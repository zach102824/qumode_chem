"""Linear H4 STO-3G Hamiltonian matching h4mol_snap_vqe.ipynb.

Geometry is H at 0, R, 2R, 3R (Å), singlet, Jordan–Wigner, 8 qubits.
Each block of 4 qubits is one qumode with Fock cutoff L=16, so the
exact qubit Hamiltonian is the 256×256 two-qumode operator used for
⟨ψ|H|ψ⟩ (same exact-H test as the H2 scripts; not the compiled
SNAP-Hadamard estimator from the notebooks).
"""

from __future__ import annotations

import numpy as np
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

NFOCK = 16
N_QUBITS = 8
N_QUMODES = 2

# Published SNAP-VQE grid: arange(0.5, 2.6, 0.1) from h4mol_params.ipynb
PAPER_GRID_A = np.arange(0.5, 2.6, 0.1)
PAPER_FCI = np.array(
    [
        -1.65311695, -1.96019364, -2.10699692, -2.16756054, -2.18031661,
        -2.16638745, -2.13797053, -2.10260848, -2.06522896, -2.02907049,
        -1.99615033, -1.96756031, -1.94369204, -1.92443064, -1.90933206,
        -1.89778065, -1.88911487, -1.88271264, -1.87803704, -1.87465158,
        -1.87221599,
    ]
)
PAPER_SNAP_ND20 = np.array(
    [
        -1.6531169543899968, -1.9601936473211554, -2.1069969178094183,
        -2.16756054745618, -2.1803166184539218, -2.1663874536450387,
        -2.1379705326974108, -2.1026084875169135, -2.0652289703047186,
        -2.0290705007671095, -1.9961503324377303, -1.9675603161692516,
        -1.9436920439073408, -1.9244306420729413, -1.9093320625772758,
        -1.8977806471287584, -1.8891148745549065, -1.8827126361248174,
        -1.8780370426789084, -1.874651579384964, -1.872215990639203,
    ]
)

# Dedicated equilibrium point stored as en_dis_p88_nd20
PAPER_EQ_R = 0.88
PAPER_EQ_FCI = -2.18041017
PAPER_EQ_SNAP = -2.1804101724879494


def h4_molecule(bond_a: float):
    """Linear H4: (0) -- (R) -- (2R) -- (3R), STO-3G singlet."""
    r = float(bond_a)
    molecule = MolecularData(
        [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, r)),
            ("H", (0.0, 0.0, 2.0 * r)),
            ("H", (0.0, 0.0, 3.0 * r)),
        ],
        "sto-3g",
        1,
        0,
    )
    return run_pyscf(molecule, run_scf=1, run_fci=1)


def jw_hamiltonian(bond_a: float) -> np.ndarray:
    """Exact 256×256 JWT matrix (same OpenFermion convention as H2).

    Index k = 0..255 is two Fock-16 qumodes: n1 = k // 16, n2 = k % 16,
    i.e. qubits 0–3 → qumode 1 and qubits 4–7 → qumode 2, matching
    ``quditvec = [4, 4]`` in h4mol_snap_vqe.ipynb.
    """
    mol = h4_molecule(bond_a)
    ham = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian()))
    return np.asarray(get_sparse_operator(ham, n_qubits=N_QUBITS).todense())


def fci_energy(bond_a: float) -> float:
    return float(h4_molecule(bond_a).fci_energy)


def lowest_eig(h: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(h)))


def paper_reference(bond_a: float) -> dict:
    """Look up published FCI / SNAP-VQE D=20 numbers when the bond is on-grid."""
    r = float(bond_a)
    out = {"r": r, "paper_fci": None, "paper_snap_nd20": None}
    if np.isclose(r, PAPER_EQ_R):
        out["paper_fci"] = PAPER_EQ_FCI
        out["paper_snap_nd20"] = PAPER_EQ_SNAP
        return out
    hits = np.where(np.isclose(PAPER_GRID_A, r))[0]
    if hits.size:
        i = int(hits[0])
        out["paper_fci"] = float(PAPER_FCI[i])
        out["paper_snap_nd20"] = float(PAPER_SNAP_ND20[i])
    return out
