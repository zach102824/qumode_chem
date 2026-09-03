#!/usr/bin/env python3
"""Hybrid (ECD+SNAP) VQE scan for linear H4, layers 1..20, always ending with ECD.

This is the hybrid depth scan. The paper SNAP-only D=20 recreation is
scripts/h4_snap.py (720 parameters, no ECD).

Paper settings (h4mol_snap_vqe.ipynb / Dutta et al., JCTC 2025)
--------------------------------------------------------------
  molecule     linear H4, H at 0, R, 2R, 3R Å, STO-3G singlet
  map          Jordan–Wigner, with HOMO/LUMO grouped in qumode 2
  SNAP block   S D on each qumode, optionally then BS(β, φ)
  ECD block    two-qumode ECD-rotation (Fig. 22): ECD1 R1 then ECD2 R2
  energy       exact 256×256 JWT Hamiltonian (not compiled SNAP-Hadamard)
  optimizer    BFGS, default maxiter=2000, tol=1e-8 (paper H4 used 2000 / 1e-12)

Ansatz
------
  one layer = (ECD block + local SNAP block + optional BS)
  circuit   = [(ECD+SNAP(+BS))]^L  then ECD
  readout   = project ancilla |g⟩ and renormalize, same as H2 ECD-VQE

Default bonds: 0.88 Å (paper equilibrium, en_dis_p88) and 2.50 Å (stretched).
By default L grows 1→20 and each depth is warm-started from the previous
best parameters plus one new random hybrid layer.

  python3 scripts/h4_hybrid.py
  python3 scripts/h4_hybrid.py --no-bs
  python3 scripts/h4_hybrid.py --layers-max 3 --maxiter 50   # smoke test
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
    jw_hamiltonian_hybrid,
    lowest_eig,
    paper_reference,
)
from h4_vqe.hybrid import (
    hybrid_energy,
    hybrid_energy_and_grad,
    hybrid_grow_params,
    hybrid_random_params,
    n_params,
)

CHEM_HA = 1.6e-3
FCI_RECREATE_HA = 1e-8
BONDS_DEFAULT = [0.88, 2.50]
LABELS = {0.88: "equilibrium", 2.50: "stretched"}


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


def _safe_fun(x, h, n_layers, nfock, use_bs):
    try:
        return hybrid_energy_and_grad(x, h, n_layers, nfock, use_bs=use_bs)
    except RuntimeError:
        return 1.0e6, np.zeros_like(x, dtype=float)


def self_check(nfock: int = 4, use_bs: bool = True) -> None:
    """Small-cutoff FD check so the script fails fast if the ansatz is broken."""
    rng = np.random.default_rng(0)
    n_layers = 1
    x = hybrid_random_params(n_layers, nfock, rng, use_bs)
    dim = nfock * nfock
    h = np.diag(np.linspace(-2.0, 1.0, dim)).astype(np.complex128)
    e, g = hybrid_energy_and_grad(x, h, n_layers, nfock, use_bs=use_bs)
    worst = 0.0
    for i in sorted({0, 7, 8, 9, x.size - 1}):
        xp = x.copy()
        xp[i] += 1e-6
        fd = (
            hybrid_energy(xp, h, n_layers, nfock, use_bs=use_bs) - e
        ) / 1e-6
        rel = abs(fd - g[i]) / (abs(g[i]) + 1e-8)
        worst = max(worst, rel)
        if rel > 0.10:
            raise RuntimeError(f"grad slot {i}: fd={fd:.4e} analytic={g[i]:.4e} rel={rel:.3f}")
    print(
        f"self-check ok (nfock={nfock}, use_bs={use_bs}, "
        f"worst FD rel err {worst:.2e})",
        flush=True,
    )


def _trial_record(res: dict, e_fci: float, seed: int) -> dict:
    err = float(res["energy"] - e_fci)
    return {
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


def _print_row(r: float, rec: dict) -> None:
    print(
        f"  r={r:.2f}  L={rec['n_layers']:<2d}  npar={rec['n_params']:<4d}  "
        f"E={rec['best_energy']:.10f}  |E-FCI|={rec['best_abs_error']:.3e}  "
        f"FCI={'yes' if rec['recreates_fci'] else 'no':<3}  "
        f"chem={'yes' if rec['chemical_acc'] else 'no'}",
        flush=True,
    )


def run_geometry(
    r: float,
    h: np.ndarray,
    e_fci: float,
    layers: range,
    seeds: list[int],
    maxiter: int,
    tol: float,
    nfock: int,
    init: str,
    use_bs: bool,
    result_geo: dict,
    out_path: Path,
    full_result: dict,
) -> None:
    prev_best_x: dict[int, np.ndarray] = {}
    layer_name = "ECD+SNAP+BS" if use_bs else "ECD+SNAP"
    for n_layers in layers:
        npar = n_params(n_layers, nfock, use_bs)
        print(f"-- r={r:.2f}  ({layer_name})^{n_layers} + ECD   npar={npar} --", flush=True)
        trials = []
        for seed in seeds:
            rng = np.random.default_rng(seed + 17 * n_layers)
            if init == "grow" and n_layers > layers.start and seed in prev_best_x:
                x0 = hybrid_grow_params(
                    prev_best_x[seed], n_layers - 1, nfock, rng, use_bs
                )
            else:
                x0 = hybrid_random_params(n_layers, nfock, rng, use_bs)
            res = _optimize(
                x0,
                lambda x, hh=h, nl=n_layers: _safe_fun(
                    x, hh, nl, nfock, use_bs
                ),
                maxiter,
                tol,
            )
            rec = _trial_record(res, e_fci, seed)
            trials.append(rec)
            prev_best_x[seed] = res["x"]
            print(
                f"    seed={seed} E={res['energy']:.10f} dE={rec['error']:+.3e} "
                f"nit={res['nit']} ok={res['success']} t={res['seconds']:.1f}s",
                flush=True,
            )
        best = _best_of(trials)
        layer_rec = {
            "n_layers": n_layers,
            "label": f"({layer_name})^{n_layers} + ECD",
            "n_params": npar,
            "best_energy": best["energy"],
            "best_abs_error": best["abs_error"],
            "recreates_fci": best["recreates_fci"],
            "chemical_acc": best["chemical_acc"],
            "best_seed": best["seed"],
            "trials": trials,
        }
        result_geo["layers"][str(n_layers)] = layer_rec
        _print_row(r, layer_rec)
        _dump(out_path, full_result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonds", type=float, nargs="+", default=BONDS_DEFAULT)
    parser.add_argument("--layers-min", type=int, default=1)
    parser.add_argument("--layers-max", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--nfock", type=int, default=NFOCK)
    parser.add_argument(
        "--init",
        choices=["grow", "random"],
        default="grow",
        help="grow: warm-start L+1 from best L plus a new layer; random: fresh each L",
    )
    parser.add_argument("--skip-check", action="store_true")
    bs_group = parser.add_mutually_exclusive_group()
    bs_group.add_argument(
        "--use-bs",
        dest="use_bs",
        action="store_true",
        help="include a two-parameter beamsplitter in every hybrid layer (default)",
    )
    bs_group.add_argument(
        "--no-bs",
        dest="use_bs",
        action="store_false",
        help="use the shared-qubit ECD blocks as the only cavity connector",
    )
    parser.set_defaults(use_bs=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output (default depends on --use-bs/--no-bs)",
    )
    args = parser.parse_args()
    if args.out is None:
        suffix = "" if args.use_bs else "_no_bs"
        args.out = Path(f"data/results/h4_hybrid{suffix}.json")

    if args.layers_min < 1 or args.layers_max < args.layers_min:
        raise SystemExit("need 1 <= --layers-min <= --layers-max")

    if not args.skip_check:
        self_check(nfock=4, use_bs=args.use_bs)

    layers = range(args.layers_min, args.layers_max + 1)
    result = {
        "settings": {
            "molecule": "linear H4, H at 0,R,2R,3R Angstrom",
            "basis": "sto-3g",
            "multiplicity": 1,
            "mapping": "jordan-wigner",
            "fermionic_mode_order_new_to_openfermion": [0, 1, 6, 7, 2, 3, 4, 5],
            "cavity_spatial_orbitals": [[0, 3], [1, 2]],
            "n_qubits": 8,
            "n_qumodes": 2,
            "nfock": args.nfock,
            "bonds": [float(r) for r in args.bonds],
            "layers_min": args.layers_min,
            "layers_max": args.layers_max,
            "always_final_ecd": True,
            "use_bs": args.use_bs,
            "seeds": args.seeds,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "init": args.init,
            "optimizer": "BFGS",
            "energy": "exact 256x256 JWT Hamiltonian",
            "ansatz": (
                "[(two-mode ECD then local SNAP"
                + (" then BS" if args.use_bs else "")
                + ")]^L then two-mode ECD, project q=|g>"
            ),
            "fci_recreate_ha": FCI_RECREATE_HA,
            "chemical_acc_ha": CHEM_HA,
        },
        "geometries": {},
    }

    for r in args.bonds:
        print(f"\n=== build H4 JWT Hamiltonian at R={r:.2f} Å ===", flush=True)
        h = jw_hamiltonian_hybrid(r)
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
            "layers": {},
        }
        result["geometries"][f"{float(r):.2f}"] = geo
        run_geometry(
            float(r),
            h,
            e_diag,
            layers,
            args.seeds,
            args.maxiter,
            args.tol,
            args.nfock,
            args.init,
            args.use_bs,
            geo,
            args.out,
            result,
        )

    print("\n=== summary (best seed per layer) ===", flush=True)
    print(
        f"{'bond':>6} {'L':>3} {'ansatz':<22} {'npar':>4} {'E':>14} {'|E-FCI|':>11} {'FCI?':>5} {'chem?':>5}",
        flush=True,
    )
    for r in args.bonds:
        geo = result["geometries"][f"{float(r):.2f}"]
        for n_layers in layers:
            rec = geo["layers"][str(n_layers)]
            print(
                f"{float(r):6.2f} {n_layers:3d} {rec['label']:<22} {rec['n_params']:4d} "
                f"{rec['best_energy']:14.10f} {rec['best_abs_error']:11.3e} "
                f"{'yes' if rec['recreates_fci'] else 'no':>5} "
                f"{'yes' if rec['chemical_acc'] else 'no':>5}",
                flush=True,
            )

    _dump(args.out, result)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
