"""Linear H4 hybrid ECD+SNAP and paper SNAP-VQE, matching Dutta et al., JCTC 2025."""

from .hamiltonian import (
    NFOCK,
    N_QUBITS,
    PAPER_FCI,
    PAPER_GRID_A,
    PAPER_SNAP_ND20,
    fci_energy,
    h4_molecule,
    jw_hamiltonian,
    paper_reference,
)
from .hybrid import (
    hybrid_energy,
    hybrid_energy_and_grad,
    hybrid_grow_params,
    hybrid_random_params,
    n_params as hybrid_n_params,
)
from .snap import (
    SNAP_DEPTH,
    load_paper_xvec,
    n_params as snap_n_params,
    snap_energy,
    snap_energy_and_grad,
    snap_random_params,
    snap_state,
)

# Hybrid layer count helper kept under the historical name.
n_params = hybrid_n_params

__all__ = [
    "NFOCK",
    "N_QUBITS",
    "PAPER_FCI",
    "PAPER_GRID_A",
    "PAPER_SNAP_ND20",
    "SNAP_DEPTH",
    "fci_energy",
    "h4_molecule",
    "jw_hamiltonian",
    "paper_reference",
    "hybrid_energy",
    "hybrid_energy_and_grad",
    "hybrid_grow_params",
    "hybrid_random_params",
    "n_params",
    "hybrid_n_params",
    "snap_n_params",
    "load_paper_xvec",
    "snap_energy",
    "snap_energy_and_grad",
    "snap_random_params",
    "snap_state",
]
