#!/usr/bin/env python3
"""Paper SNAP-displacement VQE for linear H4 at D=20.

This tries to recreate the published SNAP-VQE numbers (h4mol_snap_vqe.ipynb /
h4mol_params.ipynb). The hybrid (ECD+SNAP) 1–20 layer scan is scripts/h4_hybrid.py.

Paper settings (Dutta et al., JCTC 2025)
----------------------------------------
  molecule     linear H4, H at 0, R, 2R, 3R Å, STO-3G singlet
  map          Jordan–Wigner, 8 qubits → two Fock-16 qumodes
  ansatz       [BS (S D ⊗ S D)]^D on |0,0⟩, D=20, 720 parameters
  pack         β[D], φ[D], α[D,2], θ1[D,16], θ2[D,16]
  init         BS angles ~U(0,π), α~U(-3,3), SNAP phases ~U(0,π)
  energy       exact 256×256 JWT Hamiltonian (not compiled SNAP-Hadamard)
  optimizer    BFGS, default maxiter=2000, tol=1e-8
               (paper H4 used TensorFlow BFGS, niter=2000, threshold=1e-12)

For each default bond (0.88 Å equilibrium, 2.50 Å stretched) the script
first evaluates the stored notebook Xvec on the exact Hamiltonian, then
re-optimizes from that Xvec and from random starts.

  python3 scripts/h4_snap.py
  python3 scripts/h4_snap.py --eval-only
  python3 scripts/h4_snap.py --init random --maxiter 50 --seeds 0   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize as sciopt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h4_vqe.hamiltonian import (
    NFOCK,
    fci_energy,
    jw_hamiltonian,
    lowest_eig,
    paper_reference,
)
from h4_vqe.snap import (
    SNAP_DEPTH,
    load_paper_xvec,
    n_params,
    snap_energy,
    snap_energy_and_grad,
    snap_random_params,
    snap_state,
)

CHEM_HA = 1.6e-3
FCI_RECREATE_HA = 1e-8
BONDS_DEFAULT = [0.88, 2.50]
LABELS = {0.88: "equilibrium", 2.50: "stretched"}
PAPER_NB_DEFAULT = Path("/tmp/qumode_est_paper/h4mol_params.ipynb")


def _optimize(x0: np.ndarray, fun_and_grad, maxiter: int, tol: float) -> dict:
    t0 = time.time()
    res = sciopt.minimize(
        lambda x: fun_and_grad(x),
        x0,
        method="BFGS",
        jac=True,
        tol=tol,
        options={"maxiter": maxiter, "disp": False},
    )
    return {
        "energy": float(res.fun),
        "x": res.x,
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "message": str(res.message),
        "seconds": time.time() - t0,
    }


def _safe_fun(x, h, ndepth, nfock):
    try:
        return snap_energy_and_grad(x, h, ndepth, nfock)
    except RuntimeError:
        return 1.0e6, np.zeros_like(x, dtype=float)


def self_check(nfock: int = 4, ndepth: int = 2) -> None:
    """FD check of mixed analytic/FD SNAP gradient; vacuum energy at x=0."""
    rng = np.random.default_rng(0)
    x = snap_random_params(ndepth, nfock, rng)
    dim = nfock * nfock
    h = np.diag(np.linspace(-2.0, 1.0, dim)).astype(np.complex128)
    e0 = snap_energy(np.zeros(n_params(ndepth, nfock)), h, ndepth, nfock)
    if abs(e0 - float(np.real(h[0, 0]))) > 1e-12:
        raise RuntimeError(f"vacuum SNAP energy {e0} != H00 {h[0, 0]}")
    e, g = snap_energy_and_grad(x, h, ndepth, nfock)
    npar = x.size
    slots = [0, 1, ndepth, 2 * ndepth, 4 * ndepth, npar - 1]
    worst = 0.0
    for i in slots:
        xp = x.copy()
        xp[i] += 1e-6
        fd = (snap_energy(xp, h, ndepth, nfock) - e) / 1e-6
        rel = abs(fd - g[i]) / (abs(g[i]) + 1e-8)
        worst = max(worst, rel)
        if rel > 0.10:
            raise RuntimeError(
                f"grad slot {i}: fd={fd:.4e} analytic={g[i]:.4e} rel={rel:.3f}"
            )
    print(
        f"self-check ok (ndepth={ndepth}, nfock={nfock}, worst FD rel err {worst:.2e})",
        flush=True,
    )


def _fci_overlap(psi: np.ndarray, h: np.ndarray) -> float:
    evals, evecs = np.linalg.eigh(h)
    return float(np.abs(np.vdot(evecs[:, int(np.argmin(evals))], psi)) ** 2)


def _trial_record(res: dict, e_fci: float, label: str, seed: int | None) -> dict:
    err = float(res["energy"] - e_fci)
    return {
        "label": label,
        "seed": seed,
        "energy": res["energy"],
        "error": err,
        "abs_error": abs(err),
        "chemical_acc": abs(err) < CHEM_HA,
        "recreates_fci": abs(err) < FCI_RECREATE_HA,
        "nit": res["nit"],
        "nfev": res["nfev"],
        "success": res["success"],
        "message": res["message"],
        "seconds": res["seconds"],
        "x": res["x"].tolist(),
    }


def _best_of(trials: list[dict]) -> dict:
    return min(trials, key=lambda t: t["abs_error"])


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _eval_paper_xvec(
    r: float,
    h: np.ndarray,
    e_fci: float,
    ndepth: int,
    nfock: int,
    nb_path: Path,
    paper_snap: float | None,
) -> dict | None:
    x_paper, e_reported = load_paper_xvec(r, nb_path)
    if x_paper is None:
        print(f"  no stored Xvec in {nb_path} for R={r:.2f} Å", flush=True)
        return None
    if x_paper.size != n_params(ndepth, nfock):
        print(
            f"  stored Xvec length {x_paper.size} != D={ndepth} npar="
            f"{n_params(ndepth, nfock)}; skip eval",
            flush=True,
        )
        return None
    t0 = time.time()
    psi = snap_state(x_paper, ndepth, nfock)
    energy = float(np.real(np.vdot(psi, h @ psi)))
    err = energy - e_fci
    rec = {
        "label": "stored_xvec",
        "energy": energy,
        "error": err,
        "abs_error": abs(err),
        "chemical_acc": abs(err) < CHEM_HA,
        "recreates_fci": abs(err) < FCI_RECREATE_HA,
        "paper_reported_energy": e_reported,
        "vs_paper_reported": None if e_reported is None else energy - e_reported,
        "vs_paper_table": None if paper_snap is None else energy - paper_snap,
        "fci_overlap": _fci_overlap(psi, h),
        "seconds": time.time() - t0,
        "x": x_paper.tolist(),
    }
    print(
        f"  stored Xvec  E={energy:.10f}  |E-FCI|={abs(err):.3e}  "
        f"paper_en={e_reported}  |⟨ψ|FCI⟩|²={rec['fci_overlap']:.4f}",
        flush=True,
    )
    return rec


def run_geometry(
    r: float,
    h: np.ndarray,
    e_fci: float,
    ndepth: int,
    nfock: int,
    seeds: list[int],
    maxiter: int,
    tol: float,
    init: str,
    eval_only: bool,
    nb_path: Path,
    paper_snap: float | None,
) -> dict:
    npar = n_params(ndepth, nfock)
    print(f"-- r={r:.2f}  SNAP D={ndepth}  npar={npar} --", flush=True)
    stored = _eval_paper_xvec(r, h, e_fci, ndepth, nfock, nb_path, paper_snap)
    trials: list[dict] = []
    if eval_only:
        geo = {
            "n_depth": ndepth,
            "n_params": npar,
            "stored_xvec": stored,
            "trials": trials,
        }
        if stored is not None:
            geo.update(
                {
                    "best_energy": stored["energy"],
                    "best_abs_error": stored["abs_error"],
                    "recreates_fci": stored["recreates_fci"],
                    "chemical_acc": stored["chemical_acc"],
                    "best_label": "stored_xvec",
                }
            )
        return geo

    x_paper = None if stored is None else np.asarray(stored["x"], dtype=float)
    if init in ("paper", "both") and x_paper is not None:
        print("    optimize from stored Xvec", flush=True)
        res = _optimize(
            x_paper,
            lambda x, hh=h: _safe_fun(x, hh, ndepth, nfock),
            maxiter,
            tol,
        )
        rec = _trial_record(res, e_fci, "from_stored_xvec", None)
        rec["fci_overlap"] = _fci_overlap(snap_state(res["x"], ndepth, nfock), h)
        trials.append(rec)
        print(
            f"    from_paper E={res['energy']:.10f} dE={rec['error']:+.3e} "
            f"nit={res['nit']} ok={res['success']} t={res['seconds']:.1f}s "
            f"|⟨ψ|FCI⟩|²={rec['fci_overlap']:.4f}",
            flush=True,
        )
    elif init in ("paper", "both"):
        print("    skip paper-Xvec start (no stored vector)", flush=True)

    if init in ("random", "both"):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            x0 = snap_random_params(ndepth, nfock, rng)
            res = _optimize(
                x0,
                lambda x, hh=h: _safe_fun(x, hh, ndepth, nfock),
                maxiter,
                tol,
            )
            rec = _trial_record(res, e_fci, "random", seed)
            rec["fci_overlap"] = _fci_overlap(snap_state(res["x"], ndepth, nfock), h)
            trials.append(rec)
            print(
                f"    seed={seed} E={res['energy']:.10f} dE={rec['error']:+.3e} "
                f"nit={res['nit']} ok={res['success']} t={res['seconds']:.1f}s "
                f"|⟨ψ|FCI⟩|²={rec['fci_overlap']:.4f}",
                flush=True,
            )

    best_src = trials + ([] if stored is None else [stored])
    best = _best_of(best_src) if best_src else None
    geo = {
        "n_depth": ndepth,
        "n_params": npar,
        "stored_xvec": stored,
        "trials": trials,
    }
    if best is not None:
        geo.update(
            {
                "best_energy": best["energy"],
                "best_abs_error": best["abs_error"],
                "recreates_fci": best["recreates_fci"],
                "chemical_acc": best["chemical_acc"],
                "best_label": best.get("label"),
            }
        )
    return geo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonds", type=float, nargs="+", default=BONDS_DEFAULT)
    parser.add_argument("--depth", type=int, default=SNAP_DEPTH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--nfock", type=int, default=NFOCK)
    parser.add_argument(
        "--init",
        choices=["paper", "random", "both"],
        default="both",
        help="paper: BFGS from stored Xvec; random: notebook-style starts; both: each",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="evaluate stored notebook Xvec on exact H; do not re-optimize",
    )
    parser.add_argument("--paper-nb", type=Path, default=PAPER_NB_DEFAULT)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/results/h4_snap.json"))
    args = parser.parse_args()

    if args.depth < 1:
        raise SystemExit("need --depth >= 1")

    if not args.skip_check:
        self_check(nfock=4, ndepth=2)

    result = {
        "settings": {
            "molecule": "linear H4, H at 0,R,2R,3R Angstrom",
            "basis": "sto-3g",
            "multiplicity": 1,
            "mapping": "jordan-wigner",
            "n_qubits": 8,
            "n_qumodes": 2,
            "nfock": args.nfock,
            "bonds": [float(r) for r in args.bonds],
            "depth": args.depth,
            "n_params": n_params(args.depth, args.nfock),
            "seeds": args.seeds,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "init": args.init,
            "eval_only": args.eval_only,
            "optimizer": "BFGS",
            "energy": "exact 256x256 JWT Hamiltonian",
            "ansatz": "[BS (SNAP D ⊗ SNAP D)]^D |0,0>",
            "paper_notebook": str(args.paper_nb),
            "fci_recreate_ha": FCI_RECREATE_HA,
            "chemical_acc_ha": CHEM_HA,
        },
        "geometries": {},
    }

    for r in args.bonds:
        print(f"\n=== build H4 JWT Hamiltonian at R={r:.2f} Å ===", flush=True)
        h = jw_hamiltonian(r)
        e_diag = lowest_eig(h)
        e_pyscf = fci_energy(r)
        pref = paper_reference(r)
        label = LABELS.get(float(r), "")
        print(
            f"=== r={r:.2f} Å {label}  FCI(pyscf)={e_pyscf:.10f}  "
            f"eig={e_diag:.10f}  paper_fci={pref['paper_fci']}  "
            f"paper_SNAP_D20={pref['paper_snap_nd20']} ===",
            flush=True,
        )
        if abs(e_diag - e_pyscf) > 1e-6:
            print(
                f"WARNING: JWT eig and PySCF FCI differ by {e_diag - e_pyscf:.3e}",
                flush=True,
            )
        geo = {
            "r": float(r),
            "label": label,
            "fci_pyscf": e_pyscf,
            "fci_diag": e_diag,
            "paper_fci": pref["paper_fci"],
            "paper_snap_nd20": pref["paper_snap_nd20"],
        }
        geo.update(
            run_geometry(
                float(r),
                h,
                e_diag,
                args.depth,
                args.nfock,
                args.seeds,
                args.maxiter,
                args.tol,
                args.init,
                args.eval_only,
                args.paper_nb,
                pref["paper_snap_nd20"],
            )
        )
        result["geometries"][f"{float(r):.2f}"] = geo
        _dump(args.out, result)

    print("\n=== summary ===", flush=True)
    print(
        f"{'bond':>6} {'ansatz':<22} {'npar':>4} {'E':>14} {'|E-FCI|':>11} "
        f"{'FCI?':>5} {'chem?':>5} {'best':<18}",
        flush=True,
    )
    for r in args.bonds:
        geo = result["geometries"][f"{float(r):.2f}"]
        if "best_energy" not in geo:
            print(f"{float(r):6.2f}  (no SNAP result)", flush=True)
            continue
        print(
            f"{float(r):6.2f} {'SNAP D=' + str(geo['n_depth']):<22} {geo['n_params']:4d} "
            f"{geo['best_energy']:14.10f} {geo['best_abs_error']:11.3e} "
            f"{'yes' if geo['recreates_fci'] else 'no':>5} "
            f"{'yes' if geo['chemical_acc'] else 'no':>5} "
            f"{str(geo.get('best_label')):<18}",
            flush=True,
        )

    _dump(args.out, result)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
