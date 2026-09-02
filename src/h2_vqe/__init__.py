"""H2 ECD/SNAP VQE verification matching Dutta et al., JCTC 2025."""

from .hamiltonian import (
    BOND_DISTANCES_A,
    PAPER_FCI,
    PAPER_ECD_LCU,
    PAPER_ECD_EXACT_EV,
    PAPER_SNAP,
    fci_energy,
    gvec_hamiltonian,
    openfermion_hamiltonian,
    paper_gvec,
)
from .ecd import ecd_energy, ecd_random_params, ecd_state
from .snap import snap_energy, snap_random_params, snap_state

__all__ = [
    "BOND_DISTANCES_A",
    "PAPER_FCI",
    "PAPER_ECD_LCU",
    "PAPER_ECD_EXACT_EV",
    "PAPER_SNAP",
    "fci_energy",
    "gvec_hamiltonian",
    "openfermion_hamiltonian",
    "paper_gvec",
    "ecd_energy",
    "ecd_random_params",
    "ecd_state",
    "snap_energy",
    "snap_random_params",
    "snap_state",
]
