#!/usr/bin/env python3
"""Paper-style H2 plots plus our independent rerun."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHEM = 1.6e-3


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _best_seed(block: dict) -> np.ndarray:
    best = None
    best_err = np.inf
    for rec in block["seeds"].values():
        e = np.asarray(rec["energies"], dtype=float)
        err = float(np.max(np.abs(e - np.asarray(block.get("_fci", e)))))
        # prefer lowest mean energy
        score = float(np.mean(e))
        if score < best_err:
            best_err = score
            best = e
    return best


def main() -> None:
    verify = _load_json(Path("data/results/h2_verify.json"))
    r = np.asarray(verify["distances"])
    fci = np.asarray(verify["fci_diag_gvec"])
    paper_snap = np.asarray(verify["paper_snap_published"])
    paper_ecd = np.asarray(verify["paper_ecd_lcu_published"])
    ours_snap = np.asarray(verify["paper_snap_reeval_exact_H"])
    ours_ecd = np.asarray(verify["paper_ecd_reeval_exact_H"])

    snap_reopt = None
    ecd_reopt = None
    for key in ("snap_reopt_continuation", "snap_reopt_random"):
        if key in verify:
            snap_reopt = np.asarray(verify[key]["seeds"]["0"]["energies"])
            break
    for key in ("ecd_reopt_continuation", "ecd_reopt_random"):
        if key in verify:
            ecd_reopt = np.asarray(verify[key]["seeds"]["0"]["energies"])
            break

    out = Path("figures")
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "figure.dpi": 140})

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(r, fci, color="black", lw=1.8, label="FCI / exact diagonalization")
    ax.plot(r, paper_ecd, "o", color="#1f4e79", ms=6, label="Paper ECD-VQE (LCU energy)")
    ax.plot(r, paper_snap, "x", color="#8b3a2a", ms=7, mew=1.4, label="Paper SNAP-VQE")
    ax.plot(r, ours_ecd, "s", color="#5b8fc7", ms=4, mfc="none", label="Paper ECD params, exact H")
    ax.plot(r, ours_snap, "+", color="#c47a3a", ms=8, mew=1.2, label="Paper SNAP params, exact H")
    if ecd_reopt is not None:
        ax.plot(r, ecd_reopt, "^", color="#2a6f4e", ms=4, label="Our ECD-VQE reopt")
    if snap_reopt is not None:
        ax.plot(r, snap_reopt, "d", color="#b07d0a", ms=4, label="Our SNAP-VQE reopt")
    ax.set_xlabel("H–H bond distance (Å)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("H2 STO-3G potential energy surface")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "h2_pes.png")
    fig.savefig(out / "h2_pes.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(r, np.abs(paper_ecd - fci), "o", color="#1f4e79", label="Paper ECD-VQE (LCU)")
    ax.semilogy(r, np.abs(paper_snap - fci), "x", color="#8b3a2a", label="Paper SNAP-VQE")
    ax.semilogy(r, np.abs(ours_ecd - fci), "s", color="#5b8fc7", mfc="none", label="Paper ECD params, exact H")
    ax.semilogy(r, np.abs(ours_snap - fci), "+", color="#c47a3a", label="Paper SNAP params, exact H")
    if ecd_reopt is not None:
        ax.semilogy(r, np.abs(ecd_reopt - fci), "^", color="#2a6f4e", label="Our ECD-VQE reopt")
    if snap_reopt is not None:
        ax.semilogy(r, np.abs(snap_reopt - fci), "d", color="#b07d0a", label="Our SNAP-VQE reopt")
    ax.axhline(CHEM, color="darkorange", ls="--", lw=1.4, label="Chemical accuracy (1.6 mHa)")
    ax.set_xlabel("H–H bond distance (Å)")
    ax.set_ylabel(r"$|E - E_{\mathrm{FCI}}|$ (Hartree)")
    ax.set_title("Absolute energy error vs FCI")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "h2_error.png")
    fig.savefig(out / "h2_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(r, ours_ecd - paper_ecd, "o", color="#1f4e79", label=r"$E_{\mathrm{exact\,H}}-E_{\mathrm{paper\,LCU}}$ (ECD)")
    ax.plot(r, ours_snap - paper_snap, "x", color="#8b3a2a", label=r"$E_{\mathrm{exact\,H}}-E_{\mathrm{paper}}$ (SNAP)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("H–H bond distance (Å)")
    ax.set_ylabel("Energy difference (Hartree)")
    ax.set_title("Published energies vs exact Hamiltonian of stored states")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "h2_paper_vs_exact.png")
    fig.savefig(out / "h2_paper_vs_exact.pdf")
    plt.close(fig)

    print(f"wrote plots in {out}")


if __name__ == "__main__":
    main()
