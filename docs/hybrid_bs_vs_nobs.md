# Why hybrid `--use-bs` underperforms `--no-bs` on H4

Notes from the L=5 / L=8 / L=10 hybrid ECD+SNAP scans (seed 0, scipy BFGS,
`maxiter=2000`, exact 256×256 JWT Hamiltonian). Written 2026-09-04 from the
machine-1 runs in `data/results/`.

## Setup (reminder)

One hybrid layer is `(ECD block + local SNAP)`, optionally followed by a
two-parameter beamsplitter

\[
\mathrm{BS}(\beta,\varphi)=\exp\!\Bigl[i\tfrac{\beta}{2}\bigl(e^{i\varphi}a_1^\dagger a_2+\mathrm{h.c.}\bigr)\Bigr].
\]

Circuit: `[(ECD then SNAP(+BS))]^L` then a trailing ECD, project the ancilla
onto \(|g\rangle\). Gradients are **one-sided finite difference over every
parameter** (no analytic hybrid gradient).

- `--no-bs`: ECD is the only cavity–cavity coupler (via the shared qubit).
- `--use-bs`: each SNAP block also applies the 256×256 BS mixer.

## Headline numbers (same seed)

| mode | L | R (Å) | npar | E (Ha) | \|E−FCI\| | chem? | nit | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `--no-bs` | 5 | 0.88 | 218 | −2.1743338945 | 6.08 mHa | no | 2000 | maxiter |
| `--use-bs` | 5 | 0.88 | 228 | −1.8563936815 | 324 mHa | no | 653 | precision loss |
| `--no-bs` | 8 | 0.88 | 344 | −2.1799857168 | 0.42 mHa | **yes** | 2000 | maxiter |
| `--no-bs` | 10 | 0.88 | 428 | −2.1802644710 | 0.15 mHa | **yes** | 2000 | maxiter |
| `--no-bs` | 5 | 2.50 | 218 | −1.8680356593 | 4.18 mHa | no | 206 | precision loss |
| `--no-bs` | 8 | 2.50 | 344 | −1.8695840089 | 2.63 mHa | no | 2000 | maxiter |
| `--no-bs` | 10 | 2.50 | 428 | −1.8700674554 | 2.15 mHa | no | 2000 | maxiter |
| `--no-bs` | 11 | 2.50 | 470 | −1.8707181351 | 1.50 mHa | **yes** | 2000 | maxiter |

`--use-bs` L=5 at R=2.50 was cancelled mid-run (last live ≈ iter 448,
E≈−1.7787, \|E−FCI\|≈93.5 mHa). Further hybrid work is **`--no-bs` only**.
At R=2.50, L=8/10 stayed above chemical accuracy; L=11 reached 1.50 mHa (chem yes).

Matched-iter snapshot at R=0.88, L=5:

| iter | `--no-bs` \|E−FCI\| | `--use-bs` \|E−FCI\| |
| ---: | ---: | ---: |
| 33 | 71 mHa | 1155 mHa |
| 202 | 15 mHa | 365 mHa |
| 653 | 9.0 mHa | 324 mHa |

Both BFGS paths were monotonic (zero uphill energy steps). Late-stage
`--use-bs` crawled ~2e−9 Ha/iter and stopped on precision loss still 324 mHa
above FCI; `--no-bs` kept moving and finished at 6.1 mHa.

## Why BS hurts in *this* hybrid stack

### 1. Coupler redundancy

`--no-bs` already mixes the two cavities through ECD on the shared qubit.
BS is a **second** cavity–cavity coupler. In hybrid, that interaction family
largely overlaps what ECD already generates. Extra expressivity that is
already covered tends to create gauge-like flat directions, not a better
ground-state path.

Contrast with pure SNAP-VQE (`scripts/h4_snap.py`): there is **no** ECD, so BS
is the only inter-mode link and is necessary. That is why SNAP+BS reached
near-FCI on H4, while hybrid+BS did worse than hybrid without BS.

### 2. Ill-conditioned landscape under FD-BFGS

Random init draws each BS angle from \(U(0,\pi)\), so every layer starts with
a strong mixer on the full 256-dimensional Fock space; ECD/SNAP then have to
undo or compensate. That enlarges the Hessian condition number: many
near-null directions (BS ↔ ECD tradeoffs) plus some stiff ones.

Hybrid gradients are full one-sided FD over all parameters, including the BS
angles, with no analytic BS or ECD gradient. Flat directions are estimated
especially poorly. The matched-iter gap shows the failure is already present
before wall-clock: `--use-bs` is not merely “slower, fewer steps.”

### 3. Cost makes escape harder, but is not the root cause

Each SNAP+BS layer rebuilds a 256×256 `scipy.linalg.expm`. With FD over every
parameter, that expm is paid repeatedly (~tens of seconds per BFGS iter vs
~seconds for `--no-bs`). That burns wall time before any escape from a bad
basin is possible. The primary diagnosis from the curves is still the
landscape / redundancy argument above, not the 228 vs 218 parameter count
(~5%).

### 4. Depth check: ECD-only coupling is enough

`--no-bs` L=8 and L=10 at R=0.88 both hit chemical accuracy (0.42 and
0.15 mHa). So ECD as the sole cavity coupler is **sufficient** for this
molecule at these depths. Adding BS did not unlock a missing subspace; it
made the optimizer’s problem harder.

## Bottom line

For this hybrid ECD+SNAP stack on linear H4, BS is **redundant expressivity**
plus a **worse-conditioned landscape** under one-sided FD-BFGS. Prefer
`--no-bs` for hybrid scans. Keep BS for SNAP-only ansatze where it is the
only inter-mode coupler.

## Artifacts

- `--no-bs` results: `data/results/h4_hybrid_L{5,8,10}_nobs*.json` (+ `_iters.jsonl`, `_live.json`)
- cancelled `--use-bs` L=5: `data/results/h4_hybrid_L5_bs.json` (R=0.88 finished; R=2.50 stopped)
- per-iter logs for both modes under `data/results/`
