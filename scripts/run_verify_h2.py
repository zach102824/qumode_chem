#!/usr/bin/env python3
"""Rerun H2 ECD-VQE and SNAP-VQE with the paper GitHub settings.

Settings matched to rishabdchem/qumode_est_paper:
  molecule     H2 STO-3G, JWT, bond grid 0.25..3.25 A step 0.10
  SNAP         D=4, L=16, vacuum |0>, BFGS, α~U(0,3), θ~U(0,π)
  ECD          D=9, L=16, U|0,0> then project qubit |0>, BFGS
               |β|~U(0,3), arg(β),θ,φ ~ U(0,π)
  energy used  exact 16x16 qubit Hamiltonian (paper Eq. 28), not LCU
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

from h2_vqe.ecd import ECD_DEPTH, ecd_energy, ecd_energy_and_grad, ecd_random_params, ecd_state
from h2_vqe.hamiltonian import (
    BOND_DISTANCES_A,
    PAPER_ECD_EXACT_EV,
    PAPER_ECD_LCU,
    PAPER_FCI,
    PAPER_SNAP,
    fci_energy,
    gvec_hamiltonian,
    lowest_eig,
    openfermion_hamiltonian,
    paper_gvec,
)
from h2_vqe.snap import SNAP_DEPTH, snap_energy, snap_energy_and_grad, snap_random_params, snap_state


def load_paper(path: Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def choose_encoding(x_snap: np.ndarray, x_ecd: np.ndarray, bond: float = 0.75) -> str:
    g = paper_gvec(bond)
    scores = {}
    for enc in ("lsb", "msb"):
        h = gvec_hamiltonian(g, qubit0=enc)
        e_s = snap_energy(x_snap[5], h)
        e_e = ecd_energy(x_ecd[5], h)
        scores[enc] = abs(e_s - PAPER_FCI[5]) + abs(e_e - PAPER_FCI[5])
    return min(scores, key=scores.get), scores


def validate_hamiltonian(distances: np.ndarray, n_check: int = 5) -> list[dict]:
    rows = []
    idxs = np.linspace(0, len(distances) - 1, n_check, dtype=int)
    for i in idxs:
        r = float(distances[i])
        g = paper_gvec(r)
        h_lsb = gvec_hamiltonian(g, "lsb")
        h_msb = gvec_hamiltonian(g, "msb")
        h_of = openfermion_hamiltonian(r)
        e_pyscf = fci_energy(r)
        paper_hits = np.where(np.isclose(BOND_DISTANCES_A, r))[0]
        paper_fci = float(PAPER_FCI[int(paper_hits[0])]) if paper_hits.size else None
        rows.append(
            {
                "r": r,
                "pyscf_fci": e_pyscf,
                "paper_fci": paper_fci,
                "gvec_lsb": lowest_eig(h_lsb),
                "gvec_msb": lowest_eig(h_msb),
                "openfermion_jw": lowest_eig(h_of),
                "gvec_lsb_vs_of": float(np.linalg.norm(h_lsb - h_of)),
                "gvec_msb_vs_of": float(np.linalg.norm(h_msb - h_of)),
            }
        )
    return rows


def eval_published(x_mat: np.ndarray, kind: str, hams: list[np.ndarray]) -> np.ndarray:
    energies = np.empty(len(hams))
    fn = snap_energy if kind == "snap" else ecd_energy
    for i, h in enumerate(hams):
        energies[i] = fn(x_mat[i], h)
    return energies


def _optimize(x0: np.ndarray, fun_and_grad, maxiter: int, tol: float):
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


def reopt_scan(
    kind: str,
    hams: list[np.ndarray],
    distances: np.ndarray,
    seeds: list[int],
    maxiter: int,
    tol: float,
    continuation: bool,
    x_paper: np.ndarray | None,
) -> dict:
    depth = SNAP_DEPTH if kind == "snap" else ECD_DEPTH
    rand_fn = snap_random_params if kind == "snap" else ecd_random_params
    energy_fn_builder = snap_energy_and_grad if kind == "snap" else ecd_energy_and_grad

    out = {
        "kind": kind,
        "continuation": continuation,
        "seeds": {},
    }
    for seed in seeds:
        rng = np.random.default_rng(seed)
        energies = []
        records = []
        x_prev = None
        for i, h in enumerate(hams):
            if continuation and x_prev is not None:
                x0 = x_prev
            else:
                x0 = rand_fn(depth, rng) if kind == "ecd" else rand_fn(depth, 16, rng)
            res = _optimize(x0, lambda x, hh=h: energy_fn_builder(x, hh), maxiter, tol)
            x_prev = res["x"]
            energies.append(res["energy"])
            rec = {k: v for k, v in res.items() if k != "x"}
            rec["r"] = float(distances[i])
            records.append(rec)
            print(
                f"  {kind} seed={seed} r={distances[i]:.2f} E={res['energy']:.8f} "
                f"nit={res['nit']} ok={res['success']}",
                flush=True,
            )
        e_arr = np.asarray(energies, dtype=float)
        fci_local = np.asarray([lowest_eig(h) for h in hams], dtype=float)
        out["seeds"][str(seed)] = {
            "energies": energies,
            "records": records,
            "best_vs_fci": float(np.max(np.abs(e_arr - fci_local))),
        }
    if x_paper is not None:
        paper_e = eval_published(x_paper, kind, hams)
        out["paper_params_exact_H"] = paper_e.tolist()
    return out


def occupation(psi: np.ndarray) -> np.ndarray:
    p = np.abs(psi) ** 2
    n = np.zeros(4)
    for k in range(16):
        for q in range(4):
            if (k >> q) & 1:
                n[q] += p[k]
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-npz", type=Path, default=Path("data/paper/published_h2.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/results/h2_verify.json"))
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--snap-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--ecd-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--no-reopt", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["snap", "ecd"], choices=["snap", "ecd"])
    parser.add_argument("--reopt-mode", choices=["both", "random", "continuation"], default="both")
    parser.add_argument("--subset", type=int, default=0, help="If >0, use every Nth geometry")
    parser.add_argument(
        "--bonds",
        type=float,
        nargs="+",
        default=None,
        help="Only these bond lengths in angstrom (must match the paper grid)",
    )
    args = parser.parse_args()

    paper = load_paper(args.paper_npz)
    distances = paper["distances"]
    x_snap = paper["x_snap_nd4"]
    x_ecd = paper["x_ecd_nd9"]
    fci = PAPER_FCI
    paper_snap = PAPER_SNAP
    paper_ecd = PAPER_ECD_LCU
    paper_ecd_exact = PAPER_ECD_EXACT_EV
    if args.bonds:
        idxs = []
        for r in args.bonds:
            hits = np.where(np.isclose(np.asarray(distances, dtype=float), float(r)))[0]
            if hits.size == 0:
                raise SystemExit(f"bond {r} not on paper grid {np.asarray(distances).tolist()}")
            idxs.append(int(hits[0]))
        sl = np.asarray(idxs, dtype=int)
        distances = np.asarray(distances)[sl]
        x_snap = x_snap[sl]
        x_ecd = x_ecd[sl]
        fci = fci[sl]
        paper_snap = paper_snap[sl]
        paper_ecd = paper_ecd[sl]
        paper_ecd_exact = paper_ecd_exact[sl]
    elif args.subset > 1:
        sl = slice(None, None, args.subset)
        distances = distances[sl]
        x_snap = x_snap[sl]
        x_ecd = x_ecd[sl]
        fci = fci[sl]
        paper_snap = paper_snap[sl]
        paper_ecd = paper_ecd[sl]
        paper_ecd_exact = paper_ecd_exact[sl]

    print("=== Hamiltonian check ===", flush=True)
    ham_check_r = np.asarray(distances if args.bonds else BOND_DISTANCES_A)
    ham_rows = validate_hamiltonian(ham_check_r, n_check=min(5, len(ham_check_r)))
    for row in ham_rows:
        print(row, flush=True)

    enc, enc_scores = choose_encoding(paper["x_snap_nd4"], paper["x_ecd_nd9"])
    print(f"=== Fock encoding: {enc} scores={enc_scores} ===", flush=True)

    print("=== Build Hamiltonians ===", flush=True)
    hams = [gvec_hamiltonian(paper_gvec(r), enc) for r in distances]
    fci_exact = np.array([lowest_eig(h) for h in hams])

    print("=== Evaluate published parameters on exact H ===", flush=True)
    e_snap = eval_published(x_snap, "snap", hams)
    e_ecd = eval_published(x_ecd, "ecd", hams)
    print("SNAP |E-FCI| max", float(np.max(np.abs(e_snap - fci_exact))))
    print("ECD  |E-FCI| max", float(np.max(np.abs(e_ecd - fci_exact))))
    print("SNAP |E-paper| max", float(np.max(np.abs(e_snap - paper_snap))))
    print("ECD  |E-paper_LCU| max", float(np.max(np.abs(e_ecd - paper_ecd))))
    print("ECD  |E-paper_exactEV| max", float(np.max(np.abs(e_ecd - paper_ecd_exact))))

    occ_idx = int(np.argmin(np.abs(np.asarray(distances, dtype=float) - 0.75)))
    occ_r = float(np.asarray(distances)[occ_idx])
    occ_snap = occupation(snap_state(x_snap[occ_idx]))
    occ_ecd = occupation(ecd_state(x_ecd[occ_idx]))
    print(f"SNAP occ @{occ_r:.2f}A (lsb bits)", occ_snap)
    print(f"ECD  occ @{occ_r:.2f}A (lsb bits)", occ_ecd)

    result = {
        "settings": {
            "basis": "sto-3g",
            "mapping": "jordan-wigner",
            "nfock": 16,
            "snap_depth": 4,
            "ecd_depth": 9,
            "optimizer": "BFGS",
            "tol": args.tol,
            "maxiter": args.maxiter,
            "methods": args.methods,
            "bonds": [float(r) for r in np.asarray(distances)],
            "energy": "exact Eq.28 Hamiltonian",
            "encoding": enc,
            "encoding_scores": enc_scores,
        },
        "distances": distances.tolist(),
        "fci_paper": fci.tolist(),
        "fci_diag_gvec": fci_exact.tolist(),
        "hamiltonian_checks": ham_rows,
        "paper_snap_published": paper_snap.tolist(),
        "paper_ecd_lcu_published": paper_ecd.tolist(),
        "paper_ecd_exact_ev_published": paper_ecd_exact.tolist(),
        "paper_snap_reeval_exact_H": e_snap.tolist(),
        "paper_ecd_reeval_exact_H": e_ecd.tolist(),
        "occupations_075_lsb": {"snap": occ_snap.tolist(), "ecd": occ_ecd.tolist()},
    }

    if not args.no_reopt:
        do_random = args.reopt_mode in ("both", "random")
        do_cont = args.reopt_mode in ("both", "continuation")
        if "snap" in args.methods:
            if do_random:
                print("=== SNAP reopt, random init each geometry ===", flush=True)
                result["snap_reopt_random"] = reopt_scan(
                    "snap", hams, distances, args.snap_seeds, args.maxiter, args.tol, False, x_snap
                )
            if do_cont:
                print("=== SNAP reopt, geometry continuation ===", flush=True)
                result["snap_reopt_continuation"] = reopt_scan(
                    "snap", hams, distances, args.snap_seeds, args.maxiter, args.tol, True, x_snap
                )
        if "ecd" in args.methods:
            if do_random:
                print("=== ECD reopt, random init each geometry ===", flush=True)
                result["ecd_reopt_random"] = reopt_scan(
                    "ecd", hams, distances, args.ecd_seeds, args.maxiter, args.tol, False, x_ecd
                )
            if do_cont:
                print("=== ECD reopt, geometry continuation ===", flush=True)
                result["ecd_reopt_continuation"] = reopt_scan(
                    "ecd", hams, distances, args.ecd_seeds, args.maxiter, args.tol, True, x_ecd
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
