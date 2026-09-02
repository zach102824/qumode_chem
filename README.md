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
