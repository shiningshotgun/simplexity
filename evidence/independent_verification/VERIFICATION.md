# Independent Verification Record

**Date**: 2026-11-11  
**Scope**: Independent re-execution and audit of trained checkpoints, probing analyses, paired normalized null distribution, and checkpoint resume-equivalence.

---

## 1. Scope of Independent Re-execution

An independent verifier re-executed the following analyses against the repository's saved checkpoints on 2026-11-11:

1. **Paired Normalized Null Re-run**: Executed `scripts/compute_paired_normalized_null.py` verbatim against `checkpoints/1M/RRXOR/step_1000000.pt`.
2. **Mess3 Final-Layer Linear Probe**: Evaluated the linear probe on the full 590,490-sample dataset against `checkpoints/1M/Mess3/step_1000000.pt`.
3. **Bounded Resume-Equivalence Check**: Executed a bounded resumption test from `checkpoints/1M/RRXOR/step_200000.pt` comparing 20 uninterrupted training steps against 10 steps followed by saving in the repository's checkpoint format, reloading through the repository's resume path, and executing 10 subsequent steps.

---

## 2. Checkpoint Binding

The three checkpoints evaluated match the following SHA-256 hashes, which the verifier confirmed against the original environment's `SHA256SUMS.txt`:

- `checkpoints/1M/RRXOR/step_1000000.pt`: `75f267c2be86f83278429e7b7a4e68f00bb9eea8c5de50ac75b1beff093fbc9f`
- `checkpoints/1M/Mess3/step_1000000.pt`: `3c5760a554eaacf6dd1895b95e2e23f9205fe842a7f500f750c732ba0120f6e3`
- `checkpoints/1M/RRXOR/step_200000.pt`: `0bf6a18bc13c5967b1f0970a0986859a6d1f2eecde1bcb7b6fd187cfa9c0ce0d`

---

## 3. Verification Environment

The independent verification suite ran in an external execution environment:
- **PyTorch**: 2.14.1 (CPU-only)
- **TransformerLens**: 3.9.0
- **Python**: 3.10.12
- **Operating System**: Linux

---

## 4. Paired Normalized Null Distribution (`per_iteration_paired_null.json`)

The accompanying file `per_iteration_paired_null.json` contains the per-iteration values of the **verifier's own re-run** of the paired normalized null - representing a second, independent distribution of 1,000 paired 20/80 CV splits. The original committed run in `outputs/1M/paired_normalized_null.json` stored summary statistics rather than per-iteration raw values; thus, `per_iteration_paired_null.json` does not reconstruct the original iterations, but corroborates them through an independent run.

### Measured Comparison: Verifier Run vs. Committed Distribution

| Feature Hookpoint | Metric | Committed Mean (`outputs/1M/`) | Verifier Mean (`per_iteration_paired_null.json`) | Absolute Difference |
| :--- | :--- | :---: | :---: | :---: |
| **Concat (Real)** | Raw Test MSE | 0.14533531 | 0.14533867 | 3.36e-06 |
| **Concat (Real)** | Normalized Score ($\frac{\text{MSE}}{\text{Var}}$) | 0.36178327 | 0.36179164 | 8.38e-06 |
| **Concat (Real)** | Test Target Variance | 0.40168518 | 0.40168518 | 0.00e+00 |
| **Concat (Null)** | Raw Test MSE | 0.08836608 | 0.08836554 | 5.37e-07 |
| **Concat (Null)** | Normalized Score ($\frac{\text{MSE}}{\text{Var}}$) | 0.29716268 | 0.29716100 | 1.67e-06 |
| **Concat (Null)** | Test Target Variance | 0.29720493 | 0.29720493 | 0.00e+00 |
| **Layer 3 (Real)** | Raw Test MSE | 0.17118712 | 0.17118713 | 5.47e-09 |
| **Layer 3 (Real)** | Normalized Score ($\frac{\text{MSE}}{\text{Var}}$) | 0.42617017 | 0.42617018 | 1.36e-08 |
| **Layer 3 (Real)** | Test Target Variance | 0.40168518 | 0.40168518 | 0.00e+00 |
| **Layer 3 (Null)** | Raw Test MSE | 0.13339651 | 0.13339651 | 4.04e-09 |
| **Layer 3 (Null)** | Normalized Score ($\frac{\text{MSE}}{\text{Var}}$) | 0.45052799 | 0.45052798 | 1.31e-08 |
| **Layer 3 (Null)** | Test Target Variance | 0.29720493 | 0.29720493 | 0.00e+00 |

- **Distribution Agreement**: Observed headline absolute differences between the verifier-run distribution means and committed means are up to about 8e-6 (e.g., concat real normalized mean 0.36179164 vs committed 0.36178327). All means agree approximately at 4 decimal places (rounding can cross a boundary; we report the exact measured differences rather than a blanket match claim). The maximum observed relative difference across boundary values is 4.95e-4 on one boundary maximum.
- **Noise Analysis**: These small differences are consistent with cross-build BLAS/LAPACK floating-point accumulation noise between the two distinct environments (the original macOS environment and the Linux verification environment). This is a plausible explanation, not a proven exclusive cause; no other cause was identified during verification.

---

## 5. Additional Numerical Outcomes

1. **Mess3 Final-Layer Train MSE**:
   - Committed Value: `0.006804091621009271`
   - Verifier Value: `0.00680409148020991`
   - Absolute Difference: `1.4e-10`
2. **Resume-Equivalence Check**:
   - The 20-step bounded test from `step_200000.pt` matched bitwise over all 20 steps (per-step losses, model parameters, optimizer state, and PyTorch / NumPy RNG states).

---

## 6. Evidence Boundary & Provenance

- **Evidence Boundary**: This record verifies the final models, the committed analyses, and the resume mechanism. It does not independently prove the historical continuity of the two 1,000,000-step training runs; their history remains evidenced by the repo's committed logs and metrics.
- **Provenance Statement**: This record was produced outside the original run environment and is preserved separately from `outputs/`; original outputs are not modified by it.
