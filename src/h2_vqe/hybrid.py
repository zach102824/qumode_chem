"""Hybrid (ECD + SNAP) layer ansatz.

One layer is ECD-rotation on qubit⊗qumode followed by SNAP-displacement
on the qumode (I ⊗ SNAP@D). The qubit is kept until the end, then the
trial state is the projected, renormalized qubit-|0> component — same
readout as the paper ECD ansatz.

Circuit, left to right on |0>_q |0>_c:

    [(ECD(β_j) R(θ_j, φ_j)) then (SNAP(ϑ_j) D(α_j))]^{n}
    [optional trailing ECD-rotation]
    project qubit |0> and renormalize
"""

from __future__ import annotations

import numpy as np

from .ecd import ecd_random_params, ecd_rot_op
from .hamiltonian import NFOCK
from .snap import displace, snap_disp_op, snap_random_params

_I2 = np.eye(2, dtype=np.complex128)


def n_params(n_layers: int, final_ecd: bool, nfock: int = NFOCK) -> int:
    return n_layers * (4 + 1 + nfock) + (4 if final_ecd else 0)


def unpack_params(
    x: np.ndarray, n_layers: int, final_ecd: bool, nfock: int = NFOCK
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray | None]:
    x = np.asarray(x, dtype=float).ravel()
    expect = n_params(n_layers, final_ecd, nfock)
    if x.size != expect:
        raise ValueError(f"expected {expect} params for n_layers={n_layers}, final_ecd={final_ecd}; got {x.size}")
    off = 0
    hybrid = []
    for _ in range(n_layers):
        ecd = x[off : off + 4].copy()
        off += 4
        snap = x[off : off + 1 + nfock].copy()
        off += 1 + nfock
        hybrid.append((ecd, snap))
    final = x[off : off + 4].copy() if final_ecd else None
    return hybrid, final


def pack_params(
    hybrid: list[tuple[np.ndarray, np.ndarray]],
    final_ecd: np.ndarray | None = None,
) -> np.ndarray:
    parts = []
    for ecd, snap in hybrid:
        parts.append(np.asarray(ecd, dtype=float).ravel())
        parts.append(np.asarray(snap, dtype=float).ravel())
    if final_ecd is not None:
        parts.append(np.asarray(final_ecd, dtype=float).ravel())
    return np.concatenate(parts)


def hybrid_random_params(
    n_layers: int, final_ecd: bool, nfock: int, rng: np.random.Generator
) -> np.ndarray:
    """Paper-style guesses: ECD |β|~U(0,3), angles ~U(0,π); SNAP α~U(0,3), ϑ~U(0,π)."""
    parts = []
    for _ in range(n_layers):
        parts.append(ecd_random_params(1, rng))
        parts.append(snap_random_params(1, nfock, rng))
    if final_ecd:
        parts.append(ecd_random_params(1, rng))
    return np.concatenate(parts)


def _ecd_layer(p4: np.ndarray, nfock: int) -> np.ndarray:
    beta = float(p4[0]) * np.exp(1j * float(p4[1]))
    return ecd_rot_op(beta, float(p4[2]), float(p4[3]), nfock)


def _snap_layer(ps: np.ndarray, nfock: int) -> np.ndarray:
    return np.kron(_I2, snap_disp_op(float(ps[0]), ps[1:]))


def _destroy(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n)), k=1).astype(np.complex128)


def circuit_ops(
    x: np.ndarray, n_layers: int, final_ecd: bool, nfock: int = NFOCK
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Sequence of (kind, 32x32 op, raw params) applied left-to-right on the ket."""
    hybrid, final = unpack_params(x, n_layers, final_ecd, nfock)
    ops: list[tuple[str, np.ndarray, np.ndarray]] = []
    for ecd, snap in hybrid:
        ops.append(("ecd", _ecd_layer(ecd, nfock), ecd))
        ops.append(("snap", _snap_layer(snap, nfock), snap))
    if final is not None:
        ops.append(("ecd", _ecd_layer(final, nfock), final))
    return ops


def hybrid_state(x: np.ndarray, n_layers: int, final_ecd: bool, nfock: int = NFOCK) -> np.ndarray:
    psi32 = np.zeros(2 * nfock, dtype=np.complex128)
    psi32[0] = 1.0
    for _, op, _ in circuit_ops(x, n_layers, final_ecd, nfock):
        psi32 = op @ psi32
    qumode, _ = _project_qubit0(psi32, nfock)
    return qumode


def hybrid_energy(
    x: np.ndarray, h: np.ndarray, n_layers: int, final_ecd: bool, nfock: int = NFOCK
) -> float:
    psi = hybrid_state(x, n_layers, final_ecd, nfock)
    return float(np.real(np.vdot(psi, h @ psi)))


def _project_qubit0(psi32: np.ndarray, nfock: int) -> tuple[np.ndarray, float]:
    qumode = psi32[:nfock].copy()
    nrm = float(np.linalg.norm(qumode))
    if nrm < 1e-14:
        raise RuntimeError("hybrid projection onto |0>_qubit has vanishing amplitude")
    return qumode / nrm, nrm


def hybrid_energy_and_grad(
    x: np.ndarray,
    h: np.ndarray,
    n_layers: int,
    final_ecd: bool,
    nfock: int = NFOCK,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Energy and gradient of the projected hybrid ansatz.

    ECD parameters use the same one-sided layer finite difference as
    ``ecd_energy_and_grad``. SNAP-displacement parameters are analytic.
    """
    ops = circuit_ops(x, n_layers, final_ecd, nfock)
    dim = 2 * nfock
    rights = [np.eye(dim, dtype=np.complex128)]
    for _, op, _ in ops:
        rights.append(op @ rights[-1])
    lefts = [np.eye(dim, dtype=np.complex128)]
    for _, op, _ in reversed(ops):
        lefts.append(lefts[-1] @ op)
    lefts = list(reversed(lefts))

    vac = np.zeros(dim, dtype=np.complex128)
    vac[0] = 1.0
    psi32 = rights[-1] @ vac
    psi, nrm = _project_qubit0(psi32, nfock)
    e = float(np.real(np.vdot(psi, h @ psi)))
    w16 = h @ psi

    def _projected_dpsi(dpsi32: np.ndarray) -> np.ndarray:
        d_raw = dpsi32[:nfock]
        return (d_raw - psi * np.vdot(psi, d_raw)) / nrm

    g = np.zeros_like(x, dtype=float)
    off = 0
    a = _destroy(nfock)
    gen = a.conj().T - a

    for i, (kind, op, p) in enumerate(ops):
        if kind == "ecd":
            beta = float(p[0]) * np.exp(1j * float(p[1]))
            for slot, val in enumerate(p):
                if slot == 0:
                    layer_p = ecd_rot_op((val + eps) * np.exp(1j * float(p[1])), float(p[2]), float(p[3]), nfock)
                elif slot == 1:
                    layer_p = ecd_rot_op(float(p[0]) * np.exp(1j * (val + eps)), float(p[2]), float(p[3]), nfock)
                elif slot == 2:
                    layer_p = ecd_rot_op(beta, val + eps, float(p[3]), nfock)
                else:
                    layer_p = ecd_rot_op(beta, float(p[2]), val + eps, nfock)
                d_layer = (layer_p - op) / eps
                dpsi32 = lefts[i + 1] @ d_layer @ rights[i] @ vac
                g[off + slot] = 2.0 * float(np.real(np.vdot(w16, _projected_dpsi(dpsi32))))
            off += 4
            continue

        alpha = float(p[0])
        theta = p[1:]
        d_op = displace(nfock, alpha)
        snap_diag = np.exp(1j * theta)
        snap = np.diag(snap_diag)
        d_alpha_32 = np.kron(_I2, snap @ (gen @ d_op))
        dpsi32 = lefts[i + 1] @ d_alpha_32 @ rights[i] @ vac
        g[off] = 2.0 * float(np.real(np.vdot(w16, _projected_dpsi(dpsi32))))

        mid = rights[i] @ vac
        dv0 = d_op @ mid[:nfock]
        dv1 = d_op @ mid[nfock:]
        for k in range(nfock):
            dmid = np.zeros(dim, dtype=np.complex128)
            scale = 1j * snap_diag[k]
            dmid[k] = scale * dv0[k]
            dmid[nfock + k] = scale * dv1[k]
            dpsi32 = lefts[i + 1] @ dmid
            g[off + 1 + k] = 2.0 * float(np.real(np.vdot(w16, _projected_dpsi(dpsi32))))
        off += 1 + nfock

    return e, g
