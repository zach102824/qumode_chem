#!/usr/bin/env python3
"""Optimize a hybrid (ECD+SNAP) ansatz on H2 at the two check geometries.

Ansatz
------
One layer = ECD-rotation then SNAP-displacement on the oscillator
(I ⊗ SNAP@D). Stack 1 or 2 layers, optionally finish with one extra ECD,
then project the qubit onto |0> (same readout as paper ECD-VQE).

Compared to the paper depths (SNAP D=4, ECD D=9), this asks whether a
much shallower mixed layer can still hit FCI at 0.75 Å and 2.45 Å.
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

from h2_vqe.ecd import ecd_energy, ecd_random_params
from h2_vqe.hamiltonian import NFOCK, fci_energy, gvec_hamiltonian, lowest_eig, paper_gvec
from h2_vqe.hybrid import (
    hybrid_energy,
    hybrid_energy_and_grad,
    hybrid_random_params,
    n_params,
)
from h2_vqe.snap import snap_random_params

CHEM_HA = 1.6e-3
FCI_RECREATE_HA = 1e-8
BONDS_DEFAULT = [0.75, 2.45]
LABELS = {0.75: "equilibrium", 2.45: "stretched"}


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
        "x": res.x.tolist(),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "message": str(res.message),
        "seconds": time.time() - t0,
    }


def _safe_fun(x, h, n_layers, final_ecd):
    try:
        return hybrid_energy_and_grad(x, h, n_layers, final_ecd)
    except RuntimeError:
        return 1.0e6, np.zeros_like(x, dtype=float)


def self_check(nfock: int = NFOCK) -> None:
    """Identity SNAP must reduce to a single ECD layer; grad must match FD."""
    rng = np.random.default_rng(1)
    x_ecd = ecd_random_params(1, rng)
    x = np.concatenate([x_ecd, np.zeros(1 + nfock)])
    h = np.diag(np.linspace(-2.0, 1.0, nfock)).astype(np.complex128)
    e_h = hybrid_energy(x, h, n_layers=1, final_ecd=False, nfock=nfock)
    e_e = ecd_energy(x_ecd, h, nfock)
    if abs(e_h - e_e) > 1e-10:
        raise RuntimeError(f"hybrid vs ECD mismatch: {e_h} vs {e_e}")

    x1 = hybrid_random_params(1, False, nfock, np.random.default_rng(2))
    e, g = hybrid_energy_and_grad(x1, h, 1, False, nfock)
    worst = 0.0
    for i in (0, 3, 4, 8, 12):
        xp = x1.copy()
        xp[i] += 1e-6
        fd = (hybrid_energy(xp, h, 1, False, nfock) - e) / 1e-6
        rel = abs(fd - g[i]) / (abs(g[i]) + 1e-8)
        worst = max(worst, rel)
        if rel > 0.08:
            raise RuntimeError(f"grad slot {i}: fd={fd:.4e} analytic={g[i]:.4e} rel={rel:.3f}")
    print(f"self-check ok (worst FD rel err {worst:.2e})", flush=True)


def run_variant(
    h: np.ndarray,
    e_fci: float,
    n_layers: int,
    final_ecd: bool,
    seeds: list[int],
    maxiter: int,
    tol: float,
    nfock: int,
) -> dict:
    npar = n_params(n_layers, final_ecd, nfock)
    trials = []
    best = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x0 = hybrid_random_params(n_layers, final_ecd, nfock, rng)
        res = _optimize(x0, lambda x, hh=h: _safe_fun(x, hh, n_layers, final_ecd), maxiter, tol)
        err = float(res["energy"] - e_fci)
        rec = {
            "seed": seed,
            "error": err,
            "abs_error": abs(err),
            "chemical_acc": abs(err) < CHEM_HA,
            "recreates_fci": abs(err) < FCI_RECREATE_HA,
            **{k: v for k, v in res.items() if k != "x"},
            "x": res["x"],
        }
        trials.append(rec)
        if best is None or rec["abs_error"] < best["abs_error"]:
            best = rec
        print(
            f"  layers={n_layers} final_ecd={int(final_ecd)} seed={seed} "
            f"E={res['energy']:.10f} dE={err:+.3e} nit={res['nit']} "
            f"ok={res['success']} t={res['seconds']:.1f}s",
            flush=True,
        )
    assert best is not None
    return {
        "n_layers": n_layers,
        "final_ecd": final_ecd,
        "n_params": npar,
        "label": f"(ECD+SNAP)^{n_layers}" + (" + ECD" if final_ecd else ""),
        "best_abs_error": best["abs_error"],
        "best_energy": best["energy"],
        "recreates_fci": best["recreates_fci"],
        "chemical_acc": best["chemical_acc"],
        "best_seed": best["seed"],
        "trials": trials,
    }


def run_shallow_baseline(
    kind: str,
    h: np.ndarray,
    e_fci: float,
    depth: int,
    seeds: list[int],
    maxiter: int,
    tol: float,
    nfock: int,
) -> dict:
    """Same-depth ECD-only or SNAP-only, for whether mixing helps."""
    from h2_vqe.ecd import ecd_energy_and_grad
    from h2_vqe.snap import snap_energy_and_grad

    trials = []
    best = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if kind == "snap":
            x0 = snap_random_params(depth, nfock, rng)
            fn = lambda x, hh=h: snap_energy_and_grad(x, hh, nfock)
            npar = depth * (nfock + 1)
        else:
            x0 = ecd_random_params(depth, rng)
            fn = lambda x, hh=h: ecd_energy_and_grad(x, hh, nfock)
            npar = 4 * depth
        res = _optimize(x0, fn, maxiter, tol)
        err = float(res["energy"] - e_fci)
        rec = {
            "seed": seed,
            "error": err,
            "abs_error": abs(err),
            "chemical_acc": abs(err) < CHEM_HA,
            "recreates_fci": abs(err) < FCI_RECREATE_HA,
            **{k: v for k, v in res.items() if k != "x"},
            "x": res["x"],
        }
        trials.append(rec)
        if best is None or rec["abs_error"] < best["abs_error"]:
            best = rec
        print(
            f"  baseline {kind} D={depth} seed={seed} "
            f"E={res['energy']:.10f} dE={err:+.3e} nit={res['nit']}",
            flush=True,
        )
    assert best is not None
    return {
        "kind": kind,
        "depth": depth,
        "n_params": npar,
        "best_abs_error": best["abs_error"],
        "best_energy": best["energy"],
        "recreates_fci": best["recreates_fci"],
        "chemical_acc": best["chemical_acc"],
        "best_seed": best["seed"],
        "trials": trials,
    }


def _summarize(rows: list[dict]) -> None:
    print("\n=== summary (best seed per variant) ===", flush=True)
    hdr = f"{'bond':>6} {'ansatz':<22} {'npar':>4} {'E':>14} {'|E-FCI|':>11} {'FCI?':>5} {'chem?':>5}"
    print(hdr, flush=True)
    for row in rows:
        print(
            f"{row['r']:6.2f} {row['label']:<22} {row['n_params']:4d} "
            f"{row['best_energy']:14.10f} {row['best_abs_error']:11.3e} "
            f"{'yes' if row['recreates_fci'] else 'no':>5} "
            f"{'yes' if row['chemical_acc'] else 'no':>5}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonds", type=float, nargs="+", default=BONDS_DEFAULT)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--encoding", choices=["msb", "lsb"], default="msb")
    parser.add_argument("--nfock", type=int, default=NFOCK)
    parser.add_argument(
        "--final-ecd",
        choices=["both", "yes", "no"],
        default="both",
        help="Trailing ECD after the hybrid layers (test whether it helps)",
    )
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/results/h2_hybrid.json"))
    args = parser.parse_args()

    if not args.skip_check:
        self_check(args.nfock)

    final_flags = {"both": [False, True], "yes": [True], "no": [False]}[args.final_ecd]
    summary_rows = []
    result = {
        "settings": {
            "bonds": [float(r) for r in args.bonds],
            "layers": args.layers,
            "final_ecd": args.final_ecd,
            "seeds": args.seeds,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "encoding": args.encoding,
            "nfock": args.nfock,
            "optimizer": "BFGS",
            "fci_recreate_ha": FCI_RECREATE_HA,
            "chemical_acc_ha": CHEM_HA,
            "ansatz": "(ECD-rot then I⊗SNAP-disp)^L [then optional ECD], project q=0",
        },
        "geometries": {},
    }

    for r in args.bonds:
        g = paper_gvec(r)
        h = gvec_hamiltonian(g, args.encoding)
        e_diag = lowest_eig(h)
        e_pyscf = fci_energy(r)
        label = LABELS.get(float(r), "")
        print(
            f"\n=== r={r:.2f} Å {label}  FCI(pyscf)={e_pyscf:.10f}  "
            f"eig={e_diag:.10f} ===",
            flush=True,
        )
        geo = {
            "r": float(r),
            "label": label,
            "fci_pyscf": e_pyscf,
            "fci_diag": e_diag,
            "variants": [],
            "baselines": [],
        }
        e_ref = e_diag
        for n_layers in args.layers:
            for final_ecd in final_flags:
                print(f"-- hybrid layers={n_layers} final_ecd={final_ecd} --", flush=True)
                rec = run_variant(
                    h, e_ref, n_layers, final_ecd, args.seeds, args.maxiter, args.tol, args.nfock
                )
                geo["variants"].append(rec)
                summary_rows.append(
                    {
                        "r": float(r),
                        "label": rec["label"],
                        "n_params": rec["n_params"],
                        "best_energy": rec["best_energy"],
                        "best_abs_error": rec["best_abs_error"],
                        "recreates_fci": rec["recreates_fci"],
                        "chemical_acc": rec["chemical_acc"],
                    }
                )
        if not args.no_baselines:
            for kind in ("ecd", "snap"):
                for depth in args.layers:
                    print(f"-- baseline {kind} D={depth} --", flush=True)
                    rec = run_shallow_baseline(
                        kind, h, e_ref, depth, args.seeds, args.maxiter, args.tol, args.nfock
                    )
                    geo["baselines"].append(rec)
                    summary_rows.append(
                        {
                            "r": float(r),
                            "label": f"{kind.upper()}-only D={depth}",
                            "n_params": rec["n_params"],
                            "best_energy": rec["best_energy"],
                            "best_abs_error": rec["best_abs_error"],
                            "recreates_fci": rec["recreates_fci"],
                            "chemical_acc": rec["chemical_acc"],
                        }
                    )
        result["geometries"][f"{float(r):.2f}"] = geo

    _summarize(summary_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
