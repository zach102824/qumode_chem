#!/usr/bin/env python3
"""Pull published H2 arrays from the authors' notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _exec_cells(nb_path: Path, indices: list[int]) -> dict:
    nb = json.loads(nb_path.read_text())
    ns = {"np": np}
    for i in indices:
        exec("".join(nb["cells"][i]["source"]), ns)
    return ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/tmp/qumode_est_paper"),
        help="Clone of rishabdchem/qumode_est_paper",
    )
    parser.add_argument("--out", type=Path, default=Path("data/paper"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    params = _exec_cells(args.repo / "h2mol_params.ipynb", [37, 41])
    plots = _exec_cells(args.repo / "h2mol_plots.ipynb", list(range(5, 10)))

    np.savez(
        args.out / "published_h2.npz",
        distances=plots["dis_angstrom_vqe"],
        fci=plots["fci_en_reduced"],
        ecd_lcu=plots["en_vqe_lcu_ecd_nd9"],
        ecd_exact_ev=plots["en_exact_ev_from_lcu_ecd_nd9"],
        snap=plots["en_vqe_snap_nd4"],
        x_snap_nd4=params["Xmat_snap_disp_ansatz_nd4"],
        x_ecd_nd9=params["Xmat_ecd_rot_nd9"],
    )
    print(f"wrote {args.out / 'published_h2.npz'}")
    print("snap params", params["Xmat_snap_disp_ansatz_nd4"].shape)
    print("ecd params", params["Xmat_ecd_rot_nd9"].shape)


if __name__ == "__main__":
    main()
