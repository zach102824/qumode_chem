# H2 ECD/SNAP VQE check (Dutta et al., JCTC 2025)

Independent rerun of the H2 results in
[Simulating Electronic Structure on Bosonic Quantum Computers](https://doi.org/10.1021/acs.jctc.4c01400)
using the settings in [rishabdchem/qumode_est_paper](https://github.com/rishabdchem/qumode_est_paper).

## Matched settings

| Item | Paper / GitHub |
| --- | --- |
| Molecule | H2, STO-3G, singlet |
| Map | Jordan–Wigner, 4 qubits ↔ Fock cutoff `L=16` |
| Bond grid | 0.25–3.25 Å, step 0.10 Å |
| SNAP-VQE | depth `D=4`, vacuum `\|0⟩`, BFGS |
| ECD-VQE | depth `D=9`, `U\|0,0⟩` then project qubit `\|0⟩`, BFGS |
| Init | SNAP: `α~U(0,3)`, `θ~U(0,π)`; ECD: `\|β\|~U(0,3)`, `arg(β),θ,φ~U(0,π)` |

The notebooks minimize a **compiled** Hamiltonian (SNAP Paulis at `Nd=16`, ECD LCU `Nt=15`, `Nd=10`).
This repo evaluates and reoptimizes the **exact** Eq. (28) Hamiltonian, which is the right test of
whether the stored trial states and the ansatz are actually FCI.

## Run

```bash
# needs the authors' repo cloned (already used if present at /tmp/qumode_est_paper)
python3 scripts/extract_paper_artifacts.py
python3 scripts/run_verify_h2.py
python3 scripts/run_multistart.py
python3 scripts/plot_h2_results.py
```

Outputs: `data/results/h2_verify.json`, `data/results/h2_multistart.json`, `figures/`.

## Hybrid ECD+SNAP (`--no-bs`) H4 results

Both bond lengths are on `main` under `data/results/`:

| Bond | Files | Chem-acc layer |
|------|-------|----------------|
| R=0.88 Å | `h4_hybrid_L{5,8,10}_nobs*` / `*_r088*` | L=8 (344 params), L=10 |
| R=2.50 Å | `h4_hybrid_L{5,8,10,11}_nobs_r250*` | L=11 (470 params) |

Index: [`data/results/h4_hybrid_nobs_summary.json`](data/results/h4_hybrid_nobs_summary.json)  
Writeup: [`docs/hybrid_bs_vs_nobs.md`](docs/hybrid_bs_vs_nobs.md)

