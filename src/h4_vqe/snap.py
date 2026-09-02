"""Two-qumode SNAP-displacement ansatz matching h4mol_snap_vqe.ipynb.

One layer is

    BS(β, φ) [S(θ1) D(α1) ⊗ S(θ2) D(α2)]

with BS = exp[i (β/2) (e^{iφ} a1† a2 + h.c.)], stacked D=20 times on |0,0⟩.
Packed parameters follow ``unpack_params_ansatz`` in the notebook:

    β[D], φ[D], α[D,2], θ1[D,L], θ2[D,L]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from h2_vqe.snap import displace, snap_disp_op

from .hamiltonian import NFOCK

SNAP_DEPTH = 20
_BS_PARTS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

# Notebook names for the two default check geometries
_PAPER_XVEC_KEYS = {
    0.88: "Xvec_dis_p88_nd20",
    0.50: "Xvec_dis_p50_nd20",
    0.60: "Xvec_dis_p60_nd20",
    0.70: "Xvec_dis_p70_nd20",
    0.80: "Xvec_dis_p80_nd20",
    0.90: "Xvec_dis_p90_nd20",
    2.50: "Xvec_dis_2p5_nd20",
}


def n_params(ndepth: int = SNAP_DEPTH, nfock: int = NFOCK) -> int:
    return ndepth * (4 + 2 * nfock)


def unpack_params(
    x: np.ndarray, ndepth: int = SNAP_DEPTH, nfock: int = NFOCK
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    expect = n_params(ndepth, nfock)
    if x.size != expect:
        raise ValueError(f"expected {expect} SNAP params for D={ndepth}; got {x.size}")
    beta = x[:ndepth].copy()
    phi = x[ndepth : 2 * ndepth].copy()
    alpha = x[2 * ndepth : 4 * ndepth].reshape(ndepth, 2).copy()
    theta1 = x[4 * ndepth : 4 * ndepth + ndepth * nfock].reshape(ndepth, nfock).copy()
    theta2 = x[4 * ndepth + ndepth * nfock :].reshape(ndepth, nfock).copy()
    return beta, phi, alpha, theta1, theta2


def pack_params(beta, phi, alpha, theta1, theta2) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(beta, dtype=float).ravel(),
            np.asarray(phi, dtype=float).ravel(),
            np.asarray(alpha, dtype=float).ravel(),
            np.asarray(theta1, dtype=float).ravel(),
            np.asarray(theta2, dtype=float).ravel(),
        ]
    )


def snap_random_params(ndepth: int, nfock: int, rng: np.random.Generator) -> np.ndarray:
    """Notebook guess: BS angles ~U(0,π), α~U(-3,3), SNAP phases ~U(0,π)."""
    return pack_params(
        rng.uniform(0.0, np.pi, size=ndepth),
        rng.uniform(0.0, np.pi, size=ndepth),
        rng.uniform(-3.0, 3.0, size=(ndepth, 2)),
        rng.uniform(0.0, np.pi, size=(ndepth, nfock)),
        rng.uniform(0.0, np.pi, size=(ndepth, nfock)),
    )


def paper_xvec_key(bond_a: float) -> str | None:
    r = float(bond_a)
    for key_r, name in _PAPER_XVEC_KEYS.items():
        if np.isclose(r, key_r):
            return name
    if 0.95 < r < 2.55:
        whole = int(np.floor(r + 1e-12))
        frac = int(round((r - whole) * 10.0))
        return f"Xvec_dis_{whole}p{frac}_nd20"
    return None


def load_paper_xvec(
    bond_a: float,
    nb_path: Path = Path("/tmp/qumode_est_paper/h4mol_params.ipynb"),
) -> tuple[np.ndarray | None, float | None]:
    """Return (Xvec, reported_energy) from h4mol_params.ipynb, or (None, None)."""
    key = paper_xvec_key(bond_a)
    if key is None or not nb_path.is_file():
        return None, None
    nb = json.loads(nb_path.read_text())
    ns: dict = {"np": np}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if key not in src:
            continue
        exec(src, ns)
        if ns.get(key) is not None:
            break
    x = ns.get(key)
    en_key = key.replace("Xvec_", "en_")
    en = ns.get(en_key)
    if x is None:
        return None, None
    return np.asarray(x, dtype=float), (None if en is None else float(en))


def _destroy(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n)), k=1).astype(np.complex128)


def _bs_parts(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    key = (n1, n2)
    cached = _BS_PARTS.get(key)
    if cached is not None:
        return cached
    a1 = np.kron(_destroy(n1), np.eye(n2, dtype=np.complex128))
    a2 = np.kron(np.eye(n1, dtype=np.complex128), _destroy(n2))
    parts = (a1.conj().T @ a2, a1 @ a2.conj().T)
    _BS_PARTS[key] = parts
    return parts


def beam_splitter(beta: float, phi: float, n1: int, n2: int) -> np.ndarray:
    a1d_a2, a1_a2d = _bs_parts(n1, n2)
    gen = np.exp(1j * phi) * a1d_a2 + np.exp(-1j * phi) * a1_a2d
    return expm(1j * (0.5 * float(beta)) * gen)


def _apply_s1s2(psi: np.ndarray, s1: np.ndarray, s2: np.ndarray, n1: int, n2: int) -> np.ndarray:
    ten = psi.reshape(n1, n2)
    return (s1 @ ten @ s2.T).reshape(-1)


def _layer_op(beta, phi, alpha, theta1, theta2, nfock: int):
    s1 = snap_disp_op(float(alpha[0]), theta1)
    s2 = snap_disp_op(float(alpha[1]), theta2)
    bs = beam_splitter(float(beta), float(phi), nfock, nfock)
    return bs, s1, s2


def snap_state(x: np.ndarray, ndepth: int = SNAP_DEPTH, nfock: int = NFOCK) -> np.ndarray:
    beta, phi, alpha, theta1, theta2 = unpack_params(x, ndepth, nfock)
    psi = np.zeros(nfock * nfock, dtype=np.complex128)
    psi[0] = 1.0
    for i in range(ndepth):
        bs, s1, s2 = _layer_op(beta[i], phi[i], alpha[i], theta1[i], theta2[i], nfock)
        psi = bs @ _apply_s1s2(psi, s1, s2, nfock, nfock)
    return psi


def snap_energy(x: np.ndarray, h: np.ndarray, ndepth: int = SNAP_DEPTH, nfock: int = NFOCK) -> float:
    psi = snap_state(x, ndepth, nfock)
    return float(np.real(np.vdot(psi, h @ psi)))


def snap_energy_and_grad(
    x: np.ndarray,
    h: np.ndarray,
    ndepth: int = SNAP_DEPTH,
    nfock: int = NFOCK,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Energy and gradient on the exact two-qumode Hamiltonian.

    SNAP-displacement parameters are analytic. Beamsplitter angles use a
    one-sided finite difference of that 256×256 factor only.
    """
    beta, phi, alpha, theta1, theta2 = unpack_params(x, ndepth, nfock)
    dim = nfock * nfock
    a = _destroy(nfock)
    gen = a.conj().T - a
    layers = [
        _layer_op(beta[i], phi[i], alpha[i], theta1[i], theta2[i], nfock) for i in range(ndepth)
    ]

    vac = np.zeros(dim, dtype=np.complex128)
    vac[0] = 1.0
    rights = [vac]
    for bs, s1, s2 in layers:
        rights.append(bs @ _apply_s1s2(rights[-1], s1, s2, nfock, nfock))
    psi = rights[-1]
    e = float(np.real(np.vdot(psi, h @ psi)))
    w = h @ psi

    lefts = [np.eye(dim, dtype=np.complex128)]
    for bs, s1, s2 in reversed(layers):
        u = np.kron(s1, s2)
        lefts.append(lefts[-1] @ bs @ u)
    lefts = list(reversed(lefts))

    g_beta = np.zeros(ndepth)
    g_phi = np.zeros(ndepth)
    g_alpha = np.zeros((ndepth, 2))
    g_th1 = np.zeros((ndepth, nfock))
    g_th2 = np.zeros((ndepth, nfock))

    def _dot(dpsi: np.ndarray) -> float:
        return 2.0 * float(np.real(np.vdot(w, dpsi)))

    for i in range(ndepth):
        bs, s1, s2 = layers[i]
        mid = rights[i]
        left = lefts[i + 1]
        d1 = displace(nfock, float(alpha[i, 0]))
        d2 = displace(nfock, float(alpha[i, 1]))
        snap1 = np.exp(1j * theta1[i])
        snap2 = np.exp(1j * theta2[i])
        ten = mid.reshape(nfock, nfock)

        ds1_a = np.diag(snap1) @ (gen @ d1)
        dpsi = left @ (bs @ _apply_s1s2(mid, ds1_a, s2, nfock, nfock))
        g_alpha[i, 0] = _dot(dpsi)
        ds2_a = np.diag(snap2) @ (gen @ d2)
        dpsi = left @ (bs @ _apply_s1s2(mid, s1, ds2_a, nfock, nfock))
        g_alpha[i, 1] = _dot(dpsi)

        mid1 = d1 @ ten
        for k in range(nfock):
            dten = np.zeros((nfock, nfock), dtype=np.complex128)
            dten[k] = (1j * snap1[k]) * mid1[k]
            dpsi = left @ (bs @ (dten @ s2.T).reshape(-1))
            g_th1[i, k] = _dot(dpsi)
        mid2 = ten @ d2.T
        for k in range(nfock):
            dten = np.zeros((nfock, nfock), dtype=np.complex128)
            dten[:, k] = (1j * snap2[k]) * mid2[:, k]
            dpsi = left @ (bs @ (s1 @ dten).reshape(-1))
            g_th2[i, k] = _dot(dpsi)

        kron_mid = _apply_s1s2(mid, s1, s2, nfock, nfock)
        bs_b = beam_splitter(float(beta[i]) + eps, float(phi[i]), nfock, nfock)
        g_beta[i] = _dot(left @ ((bs_b - bs) @ kron_mid) / eps)
        bs_p = beam_splitter(float(beta[i]), float(phi[i]) + eps, nfock, nfock)
        g_phi[i] = _dot(left @ ((bs_p - bs) @ kron_mid) / eps)

    return e, pack_params(g_beta, g_phi, g_alpha, g_th1, g_th2)
