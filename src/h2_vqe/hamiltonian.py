"""H2 STO-3G Hamiltonian exactly as in the paper GitHub notebooks."""

from __future__ import annotations

import numpy as np
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.ops.representations import get_tensors_from_integrals
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf

NFOCK = 16
N_QUBITS = 4

BOND_DISTANCES_A = np.array(
    [
        0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,
        1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35,
        2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.05, 3.15, 3.25,
    ]
)

# Published notebook values from h2mol_plots.ipynb
PAPER_FCI = np.array(
    [
        -0.3122699, -0.78926939, -0.9984156, -1.09262991, -1.12990478,
        -1.13711707, -1.12836188, -1.11133942, -1.09034218, -1.06792966,
        -1.04578314, -1.02505436, -1.00648693, -0.99047634, -0.97712962,
        -0.96633454, -0.95783297, -0.95128976, -0.94634974, -0.94267779,
        -0.93998171, -0.93802086, -0.93660526, -0.93558937, -0.93486413,
        -0.93434899, -0.93398498, -0.93372922, -0.9335506, -0.93342668,
        -0.93334128,
    ]
)

PAPER_ECD_LCU = np.array(
    [
        -0.31227026, -0.78926975, -0.99841595, -1.09263026, -1.12990513,
        -1.13711741, -1.12836222, -1.11133975, -1.09034251, -1.06792998,
        -1.04578347, -1.02505468, -1.00648725, -0.99047666, -0.97712994,
        -0.96633487, -0.9578333, -0.95129009, -0.94635008, -0.94267813,
        -0.93998205, -0.93802121, -0.9366056, -0.93558971, -0.93486447,
        -0.93434933, -0.93398533, -0.93372956, -0.93355094, -0.93342701,
        -0.93334161,
    ]
)

PAPER_ECD_EXACT_EV = np.array(
    [
        -0.3122699, -0.78926939, -0.9984156, -1.09262991, -1.12990478,
        -1.13711707, -1.12836188, -1.11133942, -1.09034218, -1.06792966,
        -1.04578314, -1.02505436, -1.00648693, -0.99047634, -0.97712962,
        -0.96633454, -0.95783297, -0.95128976, -0.94634974, -0.94267779,
        -0.93998171, -0.93802086, -0.93660526, -0.93558936, -0.93486413,
        -0.93434899, -0.93398498, -0.93372922, -0.9335506, -0.93342668,
        -0.93334128,
    ]
)

PAPER_SNAP = np.array(
    [
        -0.31226989, -0.78926938, -0.99841558, -1.0926299, -1.12990477,
        -1.13711706, -1.12836187, -1.11133941, -1.09034217, -1.06792966,
        -1.04578314, -1.02505436, -1.00648693, -0.99047635, -0.97712962,
        -0.96633455, -0.95783298, -0.95128977, -0.94634975, -0.9426778,
        -0.93998172, -0.93802088, -0.93660527, -0.93558938, -0.93486415,
        -0.93434901, -0.933985, -0.93372924, -0.93355062, -0.9334267,
        -0.9333413,
    ]
)

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def h2_molecule(bond_a: float):
    molecule = MolecularData(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(bond_a)))],
        "sto-3g",
        1,
        0,
    )
    return run_pyscf(molecule, run_scf=1, run_fci=1)


def paper_gvec(bond_a: float) -> np.ndarray:
    """Eight JWT coefficients from ecd_vqe_h2mol.ipynb / snap_vqe_h2mol.ipynb."""
    mol = h2_molecule(bond_a)
    h1, h2 = get_tensors_from_integrals(mol.one_body_integrals, mol.two_body_integrals)
    v2 = h2 * 2
    g = np.zeros(8, dtype=float)
    g[0] = mol.nuclear_repulsion
    g[0] += (h1[0, 0] + h1[1, 1] + h1[2, 2] + h1[3, 3]) / 2
    g[0] += (v2[0, 1, 1, 0] + v2[2, 3, 3, 2] + v2[0, 3, 3, 0] + v2[1, 2, 2, 1]) / 4
    g[0] += (v2[0, 2, 2, 0] - v2[0, 2, 0, 2] + v2[1, 3, 3, 1] - v2[1, 3, 1, 3]) / 4
    g[1] = -h1[0, 0] / 2
    g[1] -= (v2[0, 1, 1, 0] + v2[0, 3, 3, 0] + v2[0, 2, 2, 0] - v2[0, 2, 0, 2]) / 4
    g[2] = -h1[2, 2] / 2
    g[2] -= (v2[2, 3, 3, 2] + v2[1, 2, 2, 1] + v2[0, 2, 2, 0] - v2[0, 2, 0, 2]) / 4
    g[3] = v2[0, 1, 1, 0] / 4
    g[4] = (v2[0, 2, 2, 0] - v2[0, 2, 0, 2]) / 4
    g[5] = v2[0, 3, 3, 0] / 4
    g[6] = v2[2, 3, 3, 2] / 4
    g[7] = v2[0, 3, 1, 2] / 4
    return g


def _kron_word(word: str, qubit0: str) -> np.ndarray:
    """Build a 16x16 Pauli word.

    qubit0='lsb' matches OpenFermion little-endian (qubit 0 is the least
    significant bit of the Fock index). qubit0='msb' uses numpy kron(q0,...,q3).
    ``word`` is written q0-q1-q2-q3, e.g. 'ZIII'.
    """
    ops = [_PAULI[c] for c in word]
    if qubit0 == "msb":
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out
    if qubit0 != "lsb":
        raise ValueError(f"unknown qubit0={qubit0}")
    out = ops[-1]
    for op in reversed(ops[:-1]):
        out = np.kron(out, op)
    return out


def gvec_hamiltonian(gvec: np.ndarray, qubit0: str = "lsb") -> np.ndarray:
    """Eq. (28) of the paper as a 16x16 matrix."""
    g = np.asarray(gvec, dtype=float)
    h = g[0] * np.eye(NFOCK, dtype=np.complex128)
    h += g[1] * (_kron_word("ZIII", qubit0) + _kron_word("IZII", qubit0))
    h += g[2] * (_kron_word("IIZI", qubit0) + _kron_word("IIIZ", qubit0))
    h += g[3] * _kron_word("ZZII", qubit0)
    h += g[4] * (_kron_word("ZIZI", qubit0) + _kron_word("IZIZ", qubit0))
    h += g[5] * (_kron_word("ZIIZ", qubit0) + _kron_word("IZZI", qubit0))
    h += g[6] * _kron_word("IIZZ", qubit0)
    h += g[7] * (
        _kron_word("XYYX", qubit0)
        + _kron_word("YXXY", qubit0)
        - _kron_word("XXYY", qubit0)
        - _kron_word("YYXX", qubit0)
    )
    return 0.5 * (h + h.conj().T)


def openfermion_hamiltonian(bond_a: float) -> np.ndarray:
    mol = h2_molecule(bond_a)
    ham = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian()))
    return np.asarray(get_sparse_operator(ham, n_qubits=N_QUBITS).todense())


def fci_energy(bond_a: float) -> float:
    return float(h2_molecule(bond_a).fci_energy)


def lowest_eig(h: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(h)))
