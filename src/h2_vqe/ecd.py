"""ECD-rotation ansatz matching ecd_vqe_h2mol.ipynb."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .hamiltonian import NFOCK

ECD_DEPTH = 9
N_ECD_PARAMS = 4 * ECD_DEPTH


def _destroy(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n)), k=1).astype(np.complex128)


def displace(n: int, alpha: complex) -> np.ndarray:
    a = _destroy(n)
    adag = a.conj().T
    return expm(alpha * adag - np.conj(alpha) * a)


def rotation(theta: float, phi: float) -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    gen = np.cos(phi) * sx + np.sin(phi) * sy
    return expm(-1j * (theta / 2.0) * gen)


def unpack_params(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    dim = x.size // 4
    return x[:dim], x[dim:2 * dim], x[2 * dim:3 * dim], x[3 * dim:]


def pack_params(beta_mag, beta_arg, theta, phi) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(beta_mag, dtype=float).ravel(),
            np.asarray(beta_arg, dtype=float).ravel(),
            np.asarray(theta, dtype=float).ravel(),
            np.asarray(phi, dtype=float).ravel(),
        ]
    )


def ecd_rot_op(beta: complex, theta: float, phi: float, nfock: int = NFOCK) -> np.ndarray:
    p10 = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    p01 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    ecd = np.kron(p10, displace(nfock, beta / 2.0))
    ecd += np.kron(p01, displace(nfock, -beta / 2.0))
    rot = np.kron(rotation(theta, phi), np.eye(nfock, dtype=np.complex128))
    return ecd @ rot


def ecd_unitary(x: np.ndarray, nfock: int = NFOCK) -> np.ndarray:
    beta_mag, beta_arg, theta, phi = unpack_params(x)
    beta = beta_mag * np.exp(1j * beta_arg)
    u = ecd_rot_op(beta[0], theta[0], phi[0], nfock)
    for j in range(1, beta.size):
        u = ecd_rot_op(beta[j], theta[j], phi[j], nfock) @ u
    return u


def ecd_state(x: np.ndarray, nfock: int = NFOCK) -> np.ndarray:
    """Project the ECD trial state onto qubit |0> and renormalize.

    Matches ``qumode_state_from_ecd`` in ecd_vqe_h2mol.ipynb.
    """
    psi = ecd_unitary(x, nfock) @ np.eye(2 * nfock, dtype=np.complex128)[:, 0]
    qumode = psi[:nfock].copy()
    nrm = np.linalg.norm(qumode)
    if nrm < 1e-14:
        raise RuntimeError("ECD projection onto |0>_qubit has vanishing amplitude")
    return qumode / nrm


def ecd_energy(x: np.ndarray, h: np.ndarray, nfock: int = NFOCK) -> float:
    psi = ecd_state(x, nfock)
    return float(np.real(np.vdot(psi, h @ psi)))


def _project_qubit0(psi32: np.ndarray, nfock: int) -> tuple[np.ndarray, float]:
    qumode = psi32[:nfock].copy()
    nrm = float(np.linalg.norm(qumode))
    if nrm < 1e-14:
        raise RuntimeError("ECD projection onto |0>_qubit has vanishing amplitude")
    return qumode / nrm, nrm


def ecd_energy_and_grad(x: np.ndarray, h: np.ndarray, nfock: int = NFOCK, eps: float = 1e-6) -> tuple[float, np.ndarray]:
    """Gradient of the projected ECD energy by differentiating each layer."""
    beta_mag, beta_arg, theta, phi = unpack_params(x)
    ndepth = beta_mag.size
    beta = beta_mag * np.exp(1j * beta_arg)
    layers = [ecd_rot_op(beta[j], theta[j], phi[j], nfock) for j in range(ndepth)]
    dim = 2 * nfock
    rights = [np.eye(dim, dtype=np.complex128)]
    for j in range(ndepth):
        rights.append(layers[j] @ rights[-1])
    lefts = [np.eye(dim, dtype=np.complex128)]
    for j in range(ndepth - 1, -1, -1):
        lefts.append(lefts[-1] @ layers[j])
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
    for j in range(ndepth):
        for slot, val in enumerate((beta_mag[j], beta_arg[j], theta[j], phi[j])):
            dx = np.zeros(4 * ndepth)
            # rebuild only this layer at val+eps
            if slot == 0:
                layer_p = ecd_rot_op((val + eps) * np.exp(1j * beta_arg[j]), theta[j], phi[j], nfock)
            elif slot == 1:
                layer_p = ecd_rot_op(beta_mag[j] * np.exp(1j * (val + eps)), theta[j], phi[j], nfock)
            elif slot == 2:
                layer_p = ecd_rot_op(beta[j], val + eps, phi[j], nfock)
            else:
                layer_p = ecd_rot_op(beta[j], theta[j], val + eps, nfock)
            d_layer = (layer_p - layers[j]) / eps
            dpsi32 = lefts[j + 1] @ d_layer @ rights[j] @ vac
            dpsi = _projected_dpsi(dpsi32)
            g[slot * ndepth + j] = 2.0 * float(np.real(np.vdot(w16, dpsi)))
    return e, g


def ecd_random_params(depth: int, rng: np.random.Generator) -> np.ndarray:
    """Notebook guess: |β|~U(0,3), arg(β),θ,φ ~ U(0,π)."""
    return pack_params(
        rng.uniform(0.0, 3.0, size=depth),
        rng.uniform(0.0, np.pi, size=depth),
        rng.uniform(0.0, np.pi, size=depth),
        rng.uniform(0.0, np.pi, size=depth),
    )
