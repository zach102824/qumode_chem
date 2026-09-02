#!/usr/bin/env python3
"""Random-init success rate at a few H2 geometries."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize as sciopt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h2_vqe.ecd import ECD_DEPTH, ecd_energy_and_grad, ecd_random_params
from h2_vqe.hamiltonian import fci_energy, gvec_hamiltonian, paper_gvec
from h2_vqe.snap import SNAP_DEPTH, snap_energy_and_grad, snap_random_params

DISTANCES = [0.25, 0.75, 1.75]
N_TRIALS = 5
MAXITER = 400
TOL = 1e-8
CHEM = 1.6e-3


def run_kind(kind: str, r: float, h: np.ndarray, e_fci: float) -> list[dict]:
    rows = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(1000 + 17 * trial + int(r * 100))
        if kind == "snap":
            x0 = snap_random_params(SNAP_DEPTH, 16, rng)
            fn = lambda x, hh=h: snap_energy_and_grad(x, hh)
        else:
            x0 = ecd_random_params(ECD_DEPTH, rng)
            fn = lambda x, hh=h: ecd_energy_and_grad(x, hh)
        t0 = time.time()
        res = sciopt.minimize(fn, x0, method="BFGS", jac=True, tol=TOL, options={"maxiter": MAXITER, "disp": False})
        err = float(res.fun - e_fci)
        row = {
            "kind": kind,
            "r": r,
            "trial": trial,
            "energy": float(res.fun),
            "error": err,
            "chemical_acc": abs(err) < CHEM,
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "seconds": time.time() - t0,
        }
        rows.append(row)
        print(
            f"{kind} r={r:.2f} trial={trial} E={res.fun:.8f} err={err:.3e} "
            f"chem={row['chemical_acc']} nit={res.nit}",
            flush=True,
        )
    return rows


def main() -> None:
    all_rows = []
    for r in DISTANCES:
        g = paper_gvec(r)
        h = gvec_hamiltonian(g, "msb")
        e_fci = fci_energy(r)
        print(f"=== r={r} FCI={e_fci:.8f} ===", flush=True)
        all_rows.extend(run_kind("snap", r, h, e_fci))
        all_rows.extend(run_kind("ecd", r, h, e_fci))
    out = Path("data/results/h2_multistart.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
