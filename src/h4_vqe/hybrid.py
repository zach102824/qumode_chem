"""Two-qumode hybrid (ECD + SNAP) layer for linear H4.

One layer is the paper two-qumode ECD-rotation block followed by the
paper multimode SNAP-displacement, with an optional beamsplitter. The
circuit always finishes with one extra ECD block, then the ancilla is
projected onto |g⟩ and the 256-dimensional two-qumode state is
renormalized.

Space is |q⟩ ⊗ |n1⟩ ⊗ |n2⟩, dim 2×16×16 = 512. Trial start: |g,0,0⟩.

  ECD block (paper Fig. 22 / Eq. 155), 8 parameters:
      ECD1(β1) R(θ1, φ1) on (qubit, qumode 1), then
      ECD2(β2) R(θ2, φ2) on (qubit, qumode 2)

  SNAP block (h4mol_snap_vqe.ipynb), 34 or 36 parameters:
      [S(θ1) D(α1) ⊗ I] [I ⊗ S(θ2) D(α2)]
      optionally followed by BS(β, φ)
      with BS = exp[i (β/2) (e^{iφ} a1† a2 + h.c.)]

  Circuit:  [(ECD then SNAP, optionally BS)]^L  then ECD, project qubit |0⟩
"""

from __future__ import annotations

import numpy as np

from h2_vqe.ecd import ecd_random_params, ecd_rot_op
from h2_vqe.snap import snap_disp_op

from .hamiltonian import NFOCK
from .snap import beam_splitter


def n_params(n_layers: int, nfock: int = NFOCK, use_bs: bool = True) -> int:
    """L hybrid layers plus a trailing ECD block."""
    snap_len = 2 + 2 * nfock + (2 if use_bs else 0)
    return n_layers * (8 + snap_len) + 8


def unpack_params(
    x: np.ndarray, n_layers: int, nfock: int = NFOCK, use_bs: bool = True
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    expect = n_params(n_layers, nfock, use_bs)
    if x.size != expect:
        raise ValueError(f"expected {expect} params for L={n_layers}; got {x.size}")
    snap_len = 2 + 2 * nfock + (2 if use_bs else 0)
    off = 0
    hybrid = []
    for _ in range(n_layers):
        ecd = x[off : off + 8].copy()
        off += 8
        snap = x[off : off + snap_len].copy()
        off += snap_len
        hybrid.append((ecd, snap))
    return hybrid, x[off : off + 8].copy()


def pack_params(
    hybrid: list[tuple[np.ndarray, np.ndarray]], final_ecd: np.ndarray
) -> np.ndarray:
    parts = []
    for ecd, snap in hybrid:
        parts.append(np.asarray(ecd, dtype=float).ravel())
        parts.append(np.asarray(snap, dtype=float).ravel())
    parts.append(np.asarray(final_ecd, dtype=float).ravel())
    return np.concatenate(parts)


def _ecd_block_random(rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([ecd_random_params(1, rng), ecd_random_params(1, rng)])


def _snap_block_random(
    nfock: int, rng: np.random.Generator, use_bs: bool = True
) -> np.ndarray:
    """H4 notebook guess: BS angles ~U(0,π), α~U(-3,3), SNAP phases ~U(0,π)."""
    parts = []
    if use_bs:
        parts.append(rng.uniform(0.0, np.pi, size=2))
    parts.extend(
        [
            rng.uniform(-3.0, 3.0, size=2),
            rng.uniform(0.0, np.pi, size=nfock),
            rng.uniform(0.0, np.pi, size=nfock),
        ]
    )
    return np.concatenate(parts)


def hybrid_random_params(
    n_layers: int,
    nfock: int,
    rng: np.random.Generator,
    use_bs: bool = True,
) -> np.ndarray:
    parts = []
    for _ in range(n_layers):
        parts.append(_ecd_block_random(rng))
        parts.append(_snap_block_random(nfock, rng, use_bs))
    parts.append(_ecd_block_random(rng))
    return np.concatenate(parts)


def hybrid_grow_params(
    x: np.ndarray,
    n_layers: int,
    nfock: int,
    rng: np.random.Generator,
    use_bs: bool = True,
) -> np.ndarray:
    """Pad a converged L-layer vector to L+1 by inserting one random hybrid layer."""
    hybrid, final = unpack_params(x, n_layers, nfock, use_bs)
    hybrid.append((_ecd_block_random(rng), _snap_block_random(nfock, rng, use_bs)))
    return pack_params(hybrid, final)


def _ecd_from_4(p4: np.ndarray, nfock: int) -> np.ndarray:
    beta = float(p4[0]) * np.exp(1j * float(p4[1]))
    return ecd_rot_op(beta, float(p4[2]), float(p4[3]), nfock)


def _apply_ecd_mode(psi: np.ndarray, u32: np.ndarray, mode: int, n1: int, n2: int) -> np.ndarray:
    """Apply a 2L×2L ECD-rotation on (qubit, qumode ``mode``)."""
    if mode == 1:
        u4 = u32.reshape(2, n1, 2, n1)
        ten = psi.reshape(2, n1, n2)
        return np.einsum("aibj,bjk->aik", u4, ten, optimize=True).reshape(-1)
    u4 = u32.reshape(2, n2, 2, n2)
    ten = psi.reshape(2, n1, n2)
    return np.einsum("akbl,bjl->ajk", u4, ten, optimize=True).reshape(-1)


def _apply_snap_mode(psi: np.ndarray, s: np.ndarray, mode: int, n1: int, n2: int) -> np.ndarray:
    ten = psi.reshape(2, n1, n2)
    if mode == 1:
        return np.einsum("Nn,qnj->qNj", s, ten, optimize=True).reshape(-1)
    return np.einsum("Mm,qjm->qjM", s, ten, optimize=True).reshape(-1)


def _apply_bs(psi: np.ndarray, bs: np.ndarray, n1: int, n2: int) -> np.ndarray:
    ten = psi.reshape(2, n1 * n2)
    return (ten @ bs.T).reshape(-1)


def _apply_ecd_block(psi: np.ndarray, p8: np.ndarray, n1: int, n2: int) -> np.ndarray:
    psi = _apply_ecd_mode(psi, _ecd_from_4(p8[:4], n1), 1, n1, n2)
    return _apply_ecd_mode(psi, _ecd_from_4(p8[4:], n2), 2, n1, n2)


def _apply_snap_block(
    psi: np.ndarray, ps: np.ndarray, n1: int, n2: int, use_bs: bool = True
) -> np.ndarray:
    off = 2 if use_bs else 0
    s1 = snap_disp_op(float(ps[off]), ps[off + 2 : off + 2 + n1])
    s2 = snap_disp_op(
        float(ps[off + 1]), ps[off + 2 + n1 : off + 2 + n1 + n2]
    )
    psi = _apply_snap_mode(psi, s1, 1, n1, n2)
    psi = _apply_snap_mode(psi, s2, 2, n1, n2)
    if not use_bs:
        return psi
    bs = beam_splitter(float(ps[0]), float(ps[1]), n1, n2)
    return _apply_bs(psi, bs, n1, n2)


def _vacuum(n1: int, n2: int) -> np.ndarray:
    psi = np.zeros(2 * n1 * n2, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def _project_qubit0(psi512: np.ndarray, n1: int, n2: int) -> tuple[np.ndarray, float]:
    qumode = psi512.reshape(2, n1 * n2)[0].copy()
    nrm = float(np.linalg.norm(qumode))
    if nrm < 1e-14:
        raise RuntimeError("hybrid projection onto |g>_qubit has vanishing amplitude")
    return qumode / nrm, nrm


def hybrid_state(
    x: np.ndarray, n_layers: int, nfock: int = NFOCK, use_bs: bool = True
) -> np.ndarray:
    hybrid, final = unpack_params(x, n_layers, nfock, use_bs)
    psi = _vacuum(nfock, nfock)
    for ecd, snap in hybrid:
        psi = _apply_ecd_block(psi, ecd, nfock, nfock)
        psi = _apply_snap_block(psi, snap, nfock, nfock, use_bs)
    psi = _apply_ecd_block(psi, final, nfock, nfock)
    qumode, _ = _project_qubit0(psi, nfock, nfock)
    return qumode


def hybrid_energy(
    x: np.ndarray,
    h: np.ndarray,
    n_layers: int,
    nfock: int = NFOCK,
    use_bs: bool = True,
) -> float:
    psi = hybrid_state(x, n_layers, nfock, use_bs)
    return float(np.real(np.vdot(psi, h @ psi)))


def _forward_mids(
    x: np.ndarray, n_layers: int, nfock: int, use_bs: bool
) -> tuple[list[np.ndarray], list[tuple[str, np.ndarray]]]:
    hybrid, final = unpack_params(x, n_layers, nfock, use_bs)
    steps: list[tuple[str, np.ndarray]] = []
    for ecd, snap in hybrid:
        steps.append(("ecd", ecd))
        steps.append(("snap", snap))
    steps.append(("ecd", final))
    psi = _vacuum(nfock, nfock)
    mids = [psi]
    for kind, p in steps:
        if kind == "ecd":
            psi = _apply_ecd_block(psi, p, nfock, nfock)
        else:
            psi = _apply_snap_block(psi, p, nfock, nfock, use_bs)
        mids.append(psi)
    return mids, steps


def _apply_remaining(
    psi: np.ndarray, steps, start: int, nfock: int, use_bs: bool
) -> np.ndarray:
    for kind, p in steps[start:]:
        if kind == "ecd":
            psi = _apply_ecd_block(psi, p, nfock, nfock)
        else:
            psi = _apply_snap_block(psi, p, nfock, nfock, use_bs)
    return psi


def hybrid_energy_and_grad(
    x: np.ndarray,
    h: np.ndarray,
    n_layers: int,
    nfock: int = NFOCK,
    eps: float = 1e-6,
    use_bs: bool = True,
) -> tuple[float, np.ndarray]:
    """Projected energy and one-sided finite-difference gradient."""
    mids, steps = _forward_mids(x, n_layers, nfock, use_bs)
    psi, _nrm = _project_qubit0(mids[-1], nfock, nfock)
    e = float(np.real(np.vdot(psi, h @ psi)))

    def _energy_from(psi512: np.ndarray) -> float:
        q, n = _project_qubit0(psi512, nfock, nfock)
        return float(np.real(np.vdot(q, h @ q)))

    g = np.zeros_like(x, dtype=float)
    off = 0
    for i, (kind, p) in enumerate(steps):
        base = mids[i]
        for slot in range(p.size):
            pp = p.copy()
            pp[slot] += eps
            if kind == "ecd":
                psi_p = _apply_ecd_block(base, pp, nfock, nfock)
            else:
                psi_p = _apply_snap_block(base, pp, nfock, nfock, use_bs)
            psi_p = _apply_remaining(psi_p, steps, i + 1, nfock, use_bs)
            g[off + slot] = (_energy_from(psi_p) - e) / eps
        off += p.size
    return e, g
