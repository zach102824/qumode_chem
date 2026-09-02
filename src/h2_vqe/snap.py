"""SNAP-displacement ansatz matching snap_vqe_h2mol.ipynb."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .hamiltonian import NFOCK

SNAP_DEPTH = 4
N_SNAP_PARAMS = SNAP_DEPTH * (NFOCK + 1)


def _destroy(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n)), k=1).astype(np.complex128)


def displace(n: int, alpha: float) -> np.ndarray:
    a = _destroy(n)
    adag = a.conj().T
    return expm(alpha * adag - np.conj(alpha) * a)


def unpack_params(x: np.ndarray, nfock: int = NFOCK) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    ndepth = x.size // (nfock + 1)
    alpha = x[:ndepth].copy()
    theta = x[ndepth:ndepth + ndepth * nfock].reshape(ndepth, nfock)
    return alpha, theta


def pack_params(alpha: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(alpha, dtype=float).ravel(), np.asarray(theta, dtype=float).ravel()])


def snap_disp_op(alpha: float, thetavec: np.ndarray) -> np.ndarray:
    nfock = thetavec.shape[0]
    snap = np.diag(np.exp(1j * thetavec))
    return snap @ displace(nfock, float(alpha))


def snap_unitary(x: np.ndarray, nfock: int = NFOCK) -> np.ndarray:
    alpha, theta = unpack_params(x, nfock)
    u = snap_disp_op(alpha[0], theta[0])
    for i in range(1, alpha.size):
        u = snap_disp_op(alpha[i], theta[i]) @ u
    return u


def snap_state(x: np.ndarray, nfock: int = NFOCK) -> np.ndarray:
    vac = np.zeros(nfock, dtype=np.complex128)
    vac[0] = 1.0
    return snap_unitary(x, nfock) @ vac


def snap_energy(x: np.ndarray, h: np.ndarray, nfock: int = NFOCK) -> float:
    psi = snap_state(x, nfock)
    return float(np.real(np.vdot(psi, h @ psi)))


def snap_energy_and_grad(x: np.ndarray, h: np.ndarray, nfock: int = NFOCK) -> tuple[float, np.ndarray]:
    """Analytic gradient of ⟨ψ|H|ψ⟩ for the SNAP-displacement ansatz."""
    alpha, theta = unpack_params(x, nfock)
    ndepth = alpha.size
    a = _destroy(nfock)
    gen = a.conj().T - a
    layers = [snap_disp_op(alpha[i], theta[i]) for i in range(ndepth)]
    rights = [np.eye(nfock, dtype=np.complex128)]
    for i in range(ndepth):
        rights.append(layers[i] @ rights[-1])
    u = rights[-1]
    lefts = [np.eye(nfock, dtype=np.complex128)]
    for i in range(ndepth - 1, -1, -1):
        lefts.append(lefts[-1] @ layers[i])
    lefts = list(reversed(lefts))

    vac = np.zeros(nfock, dtype=np.complex128)
    vac[0] = 1.0
    psi = u @ vac
    e = float(np.real(np.vdot(psi, h @ psi)))
    w = h @ psi

    g_alpha = np.zeros(ndepth)
    g_theta = np.zeros((ndepth, nfock))
    for i in range(ndepth):
        d_op = displace(nfock, float(alpha[i]))
        snap_diag = np.exp(1j * theta[i])
        snap = np.diag(snap_diag)
        d_layer_alpha = snap @ (gen @ d_op)
        dpsi = lefts[i + 1] @ d_layer_alpha @ rights[i] @ vac
        g_alpha[i] = 2.0 * float(np.real(np.vdot(w, dpsi)))
        mid = d_op @ (rights[i] @ vac)
        scale = 1j * snap_diag * mid
        dpsi_th = lefts[i + 1] * scale[None, :]
        g_theta[i] = 2.0 * np.real(np.conj(w) @ dpsi_th)
    return e, pack_params(g_alpha, g_theta)


def snap_random_params(depth: int, nfock: int, rng: np.random.Generator) -> np.ndarray:
    """Notebook guess: α~U(0,3), θ~U(0,π)."""
    return pack_params(rng.uniform(0.0, 3.0, size=depth), rng.uniform(0.0, np.pi, size=(depth, nfock)))
