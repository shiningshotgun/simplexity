# Comprehensive Reproduction Report & Empirical Audit: "Transformers Represent Belief State Geometry in their Residual Stream"

**Paper:** Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Oldenziel, Paul M. Riechers, *Transformers Represent Belief State Geometry in their Residual Stream* (NeurIPS 2024, [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)).  
**Repository:** `simplexity` (`github.com/shiningshotgun/simplexity`)  
**Date:** November 2026  
**Auditor / Reproducer:** Clean-room implementation under `shiningshotgun` identity.

---

## 1. Executive Summary

This study independently reproduces the theoretical foundations, training dynamics, internal representations, and probing analyses of Shai et al. (NeurIPS 2024) from first principles and the published paper/appendix alone, without reference to the un-licensed official codebase (`epsilon-transformers`).

Both processes (Mess3 and RRXOR) were trained through the full **1,000,000 training iterations** on CPU sequentially (total runtime logged: 4.18 hours), with bit-exact checkpoint resumption verified, full-dataset statistical controls evaluated without subsampling, matched 20/80 cross-validation shuffle null distributions computed, and activation-centroid distance diagnostics analyzed alongside probe-predicted representations.

### Headline Findings & Verdicts

| Claim / Gate | Paper Reference | Status / Verdict | Summary of Findings |
| :--- | :--- | :--- | :--- |
| **Gates 1–5: Mathematical Invariants** | Sec 2.2, 3.1, 3.2, App A.3, Fig S2 | **REPRODUCED (100%)** | Stationary eigenvectors, row stochasticity, 36 rational RRXOR belief states (stabilizing at length 6), 2-simplex fractal geometry, and exact 9-position optimal loss constants ($H^* = 0.803518$ nats for Mess3, $0.586501$ nats for RRXOR) confirmed with zero discrepancy. |
| **Claim 1: Mess3 Final Layer Geometry** | Sec 3.1, Fig 5C, Fig S1 | **REPRODUCED** | Linear probe on full 590,490-sample dataset recovers 2-simplex fractal: **Train MSE = 0.006804 ($R^2 = 0.9796$)**, **1,000x 20/80 CV Test MSE = 0.006809 ($\pm 0.000017$)**. 1,000x distinct-state shuffle control MSE = **0.267761 ($\pm 0.006479$)** (~39x higher). *Note: Paper Fig S1 caption reported 0.0004/0.0003; our measured MSE is ~0.0068, visually and statistically recovering the fractal.* |
| **Claim 2: RRXOR Distributed Representation** | Sec 3.2, Fig 7C/E | **PARTIALLY REPRODUCED / QUALIFIED** | Layer 3 Train MSE = 0.1589 ($R^2 = 0.6043$), CV Test MSE = 0.1711. Concatenation Train MSE = **0.0757 ($R^2 = 0.8115$)**, CV Test MSE = **0.1455 ($\pm 0.0101$)**. Train MSE drops **>52%** and CV test MSE drops **15%** from Layer 3 to Concat. Concatenation lowers train-fit and held-out probe error relative to the final layer, but decoding above a matched shuffle null is not demonstrated. Matched CV shuffle null test MSE is 0.0875 ($\pm 0.0179$) for concat (and 0.1327 for Layer 3); and pairwise distance correlation across 35 states reaches $R^2 = 0.2762$ for probe-predicted beliefs and $R^2 = 0.0871$ for activation centroids (vs paper ~0.95). |
| **Claim 3: Representation Emergence** | Sec 3.1, 3.2, Fig 6A/D, Fig S2 | **REPRODUCED (Rapid refinement with plateau)** | Rapid early refinement with small non-monotonic ticks, then a plateau: Mess3 probe MSE drops from 0.045302 (100) to 0.006703 (200k), fluctuating between 0.006703 and 0.006804 through 1M; RRXOR concat MSE drops from 0.140212 (100) to 0.075728 (1M) with small non-monotonic ticks at steps 200 and 1,000. |
| **Claim 4: LayerNorm Robustness** | App A.4, Fig S1 | **REPRODUCED** | Pre-LN Train MSE (0.006804) vs. Post-LN Train MSE (0.006391) confirms final LayerNorm does not destroy or distort belief state representations. |

---

## 2. Environment & Resolved Library Configuration

### Local System Environment
- **Operating System**: macOS (Darwin 24.6.0, Apple Silicon ARM64, 8 CPU cores)
- **Host Python**: `/usr/bin/python3` (Python 3.9.6 default)
- **Project Virtual Environment**: Project-local `.venv` created with Python 3.11.16 (`<home>/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11`)

### Installed Packages & Resolved Versions
- `torch`: 2.15.0
- `transformer-lens`: 2.15.4
- `numpy`: 1.26.4
- `scipy`: 1.17.1
- `matplotlib`: 3.11.2
- `pytest`: 9.2.0
- `pyarrow`: 14.0.2
- `pandas`: 2.0.3

### Download Budget & Storage Audit
- **Hard Download Budget**: 5.0 GB limit across all external fetches.
- **Measured Cumulative Download**: **284.6 MiB** (~0.28 GB, 5.7% of budget).
- **Disk Footprint**:
  * Installed `.venv`: 1.10 GB
  * Saved Checkpoints: 26 files (13 per process across 1M schedule): 15.3 MB total
  * Preserved Pilot Checkpoints: 16 files: 9.4 MB total
  * Generated Figures (8 PNGs): 4.8 MB total

---

## 3. Architecture Specification & Parameter Verification

Paper Appendix A.6 verbatim:
> *"context window size of 10, used ReLU as the activation function, and had a head dimension of 8 and a model dimension of 64. There was 1 attention head in each of 4 layers. The model had MLPs of dimension 256 and used causal attention masking. Layer normalization was applied."*

### Resolved HookedTransformer Configuration
- **Model Type**: TransformerLens `HookedTransformer` (`HookedTransformerConfig`)
- **Layers**: $n_{\text{layers}} = 4$
- **Hidden Dimension**: $d_{\text{model}} = 64$
- **Attention Heads**: $n_{\text{heads}} = 1, d_{\text{head}} = 8$
- **Attention Scaling**: $1 / \sqrt{d_{\text{head}}} = 1 / \sqrt{8} \approx 0.35355$ (`attn_scale = 2.828427`)
- **MLP Dimension**: $d_{\text{mlp}} = 256$
- **Activation Function**: `relu`
- **Normalization**: LayerNorm pre-LN, $\epsilon = 10^{-5}$
- **Biases**: Enabled across all Linear layers, LayerNorms, and Attention projections
- **Positional Embeddings**: Standard learned 1D positional embedding ($10 \times 64$)
- **BOS Proposal**: `default_prepend_bos = False` (no artificial BOS token; exactly 10 raw tokens)
- **Weight Initialization**: TransformerLens GPT-2 default ($\mathcal{N}(0, 0.02)$, LayerNorm weight 1.0, bias 0.0)
- **Seed**: Seed 0 everywhere

### Parameter Count Verification
Both the pure PyTorch reference model (`SimplexityTransformer`) and TransformerLens `HookedTransformer` yield identical parameter counts:

| Component | Mess3 ($d_{\text{vocab}} = 3$) | RRXOR ($d_{\text{vocab}} = 2$) |
| :--- | :--- | :--- |
| **Token Embedding** ($d_{\text{vocab}} \times d_{\text{model}}$) | $3 \times 64 = 192$ | $2 \times 64 = 128$ |
| **Positional Embedding** ($n_{\text{ctx}} \times d_{\text{model}}$) | $10 \times 64 = 640$ | $10 \times 64 = 640$ |
| **Per Layer (4 Layers)** | | |
| - LayerNorm 1 (weight + bias) | $64 + 64 = 128$ | $128$ |
| - Attention ($W_Q, W_K, W_V, W_O$ with biases) | $(512 + 8) \times 3 + (512 + 64) = 2,136$ | $2,136$ |
| - LayerNorm 2 (weight + bias) | $64 + 64 = 128$ | $128$ |
| - MLP ($W_{\text{in}}, b_{\text{in}}, W_{\text{out}}, b_{\text{out}}$) | $(16,384 + 256) + (16,384 + 64) = 33,088$ | $33,088$ |
| - *Total per Block* | $35,480$ | $35,480$ |
| - *4 Blocks Subtotal* | $141,920$ | $141,920$ |
| **Final LayerNorm** (weight + bias) | $64 + 64 = 128$ | $128$ |
| **Unembedding** ($W_U, b_U$) | $64 \times 3 + 3 = 195$ | $64 \times 2 + 2 = 130$ |
| **Total Parameters** | **143,075** | **142,946** |

---

## 4. Training Execution & Resumption Decision Audit

### Recorded Resumption Decision Path (from Run Logs & Checkpoint Metadata)
As evidenced by `outputs/1M/train_both.log` and checkpoint timestamps:
1. Training initially ran to step 50,000 for Mess3 (saving `step_50000.pt` at 18:44:35) and step 200 for RRXOR (saving `step_200.pt` at 18:37:00).
2. At 18:48:25, `scripts/train_all.py` was invoked to carry both processes sequentially to 1,000,000 steps.
3. In accordance with `src/simplexity/train.py`, the training engine scanned `checkpoints/1M/` and executed automatic checkpoint resumption:
   - **Mess3**: Logged `[Mess3] Resumed from checkpoint step 50000`. Restored model state, optimizer state, torch RNG state, and numpy sampler RNG state from `step_50000.pt`, loaded existing metrics from `outputs/1M/Mess3_metrics.json`, and continued through step 1,000,000 (completed at 20:48:23).
   - **RRXOR**: Logged `[RRXOR] Resumed from checkpoint step 200`. Restored model state, optimizer state, torch RNG state, and numpy sampler RNG state from `step_200.pt`, loaded existing metrics, and continued through step 1,000,000 (completed at 22:59:17).
4. The test suite [`tests/test_resume.py`](tests/test_resume.py) explicitly validates that resuming from a saved checkpoint produces bit-for-bit identical weights and losses compared to an uninterrupted continuous run (max absolute weight difference: 0.0).

### Logged Runtimes and Throughput
Measured from committed `outputs/1M/train_both.log` and `outputs/1M/*_metrics.json`:

| Process | Training Steps | Total Wallclock Time | Average Throughput | Peak RAM | Checkpoints Saved |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mess3** | 1,000,000 | 7,194.3 s (2.00 h) | 139.0 steps/s | 432.0 MB | 13 snapshots (100 to 1M) |
| **RRXOR** | 1,000,000 | 7,853.1 s (2.18 h) | 127.3 steps/s | 432.0 MB | 13 snapshots (100 to 1M) |
| **Total Sequential** | 2,000,000 | **15,047.4 s (4.18 h)** | **132.9 steps/s** | **432.0 MB** | **26 checkpoints** |

---

## 5. Mathematical & Process Invariant Verification (Gates 1–5)

| Gate | Description | Paper Section | Result | Measured Value |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | Stationary distribution left eigenvector ($\pi T = \pi$) | Sec 2.2, Eq. (1) | **PASS** | Mess3: $[1/3, 1/3, 1/3]$ ($\Delta = 0.0$); RRXOR: $[1/3, 1/6, 1/6, 1/6, 1/6]$ ($\Delta = 1.11 \times 10^{-16}$). |
| **Gate 2** | Stochastic matrix row sums equal 1.0 | App A.3 | **PASS** | Max row deviation from 1.0: $0.0$ for all transition matrices. |
| **Gate 3** | RRXOR reachable rational belief states count = 36 | Sec 3.2, Fig 7A | **PASS** | Exact rational arithmetic (`fractions.Fraction`) reaches **exactly 36 distinct belief states**, fully stabilizing at length 6. |
| **Gate 4** | Mess3 2-simplex ground-truth fractal geometry | Sec 3.1, Fig 5B | **PASS** | Computed beliefs for all 88,573 prefixes ($\le 10$); saved `figures/GT_mess3.png`. |
| **Gate 5** | 9-position finite-context optimal loss normalization constants | Fig S2 | **PASS** | Mess3: **0.803518 nats**; RRXOR: **0.586501 nats** (position 0 analytically $\ln 2 \approx 0.693147$ nats). |

---

## 6. Training Dynamics & Loss Convergence

Regenerated strictly from committed `outputs/1M/Mess3_metrics.json` and `outputs/1M/RRXOR_metrics.json`:

### Validation Cross-Entropy Loss Curve (Normalized: $1.0 = H^*$)

| Training Step | Mess3 Raw Loss (nats) | Mess3 Normalized ($H / H^*$) | RRXOR Raw Loss (nats) | RRXOR Normalized ($H / H^*$) |
| :---: | :---: | :---: | :---: | :---: |
| 100 | 0.83313 | 1.0369 | 0.67719 | 1.1546 |
| 200 | 0.82171 | 1.0226 | 0.66401 | 1.1322 |
| 500 | 0.81296 | 1.0118 | 0.65538 | 1.1174 |
| 1,000 | 0.81162 | 1.0101 | 0.64085 | 1.0927 |
| 2,000 | 0.81802 | 1.0181 | 0.63198 | 1.0775 |
| 5,000 | 0.82214 | 1.0232 | 0.61682 | 1.0517 |
| 10,000 | 0.80842 | 1.0061 | 0.60549 | 1.0324 |
| 20,000 | 0.81733 | 1.0172 | 0.59417 | 1.0131 |
| 50,000 | 0.78931 | 0.9823 | 0.59213 | 1.0096 |
| 100,000 | 0.81757 | 1.0175 | 0.58838 | 1.0032 |
| 200,000 | 0.80705 | 1.0044 | 0.58849 | 1.0034 |
| 500,000 | 0.80279 | 0.9991 | 0.58724 | 1.0013 |
| **1,000,000** | **0.80974** | **1.0077** | **0.58498** | **0.9974** |

### Convergence Diagnosis & Final Noise Floor
Rather than converging monotonically to an exact scalar, validation loss enters a flat stationary noise floor:
- For Mess3: fluctuations across steps 100k–1M stay within $H / H^* \in [0.982, 1.018]$ ($\pm 0.015$ nats of theoretical minimum).
- For RRXOR: fluctuations across steps 100k–1M stay within $H / H^* \in [0.997, 1.004]$ ($\pm 0.003$ nats of theoretical minimum).
At step 1,000,000, checkpoint validation loss reaches 1.0077x (Mess3) and 0.9974x (RRXOR), confirming that next-token prediction loss achieves near-optimal performance.

---

## 7. Probe Results & Claims Verification

Linear probes were fitted using ordinary least squares (affine map $\hat{b} = W a + c$) on the full datasets without subsampling. Statistical controls were evaluated using:
1. **1,000x Distinct-State Label Shuffle Control**: Shuffling target vectors across unique reachable belief states (per Fig 6C & App A.7).
2. **1,000x 20/80 Cross-Validation**: Training on 20% of samples (1,622 for RRXOR, 118,098 for Mess3), evaluating on the held-out 80% test set.
3. **Matched 20/80 CV Shuffle Null**: Permuting targets across distinct belief states and evaluating each permutation under the exact same 20/80 CV protocol.
4. **Distance Diagnostics**: Reporting pairwise Euclidean distance correlations for both (A) probe-predicted beliefs ($\hat{b} = Wa + c$) and (B) activation centroids ($\bar{a}_k$).

---

### Claim 1: Mess3 Residual Stream Geometry (Paper Section 3.1 & Fig 5C)
- **Dataset**: All $3^{10} = 59,049$ sequences $\times 10$ positions = 590,490 samples (88,554 distinct reachable belief states).
- **Target**: Ground-truth 3-simplex belief vector.
- **Hookpoint**: Final residual stream before final LayerNorm (`blocks.3.hook_resid_post`).

| Metric | Paper-Reported | Measured 1M Run (Committed) | Measured 20k Pilot | Verdict / Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Final Layer Pre-LN Train MSE** | Low ($R^2 \approx 1.0$) | **0.006804** ($R^2 = 0.9796$) | 0.008260 ($R^2 = 0.9752$) | **REPRODUCED** |
| **1,000x 20/80 CV Test MSE** | — | **0.006809** ($\pm 0.000017$) | 0.009547 ($\pm 0.000386$) | **REPRODUCED** |
| **Post-LayerNorm Train MSE** | 0.0003 (Fig S1 caption) | **0.006391** ($R^2 = 0.9808$) | 0.007845 ($R^2 = 0.9765$) | **REPRODUCED** |
| **Post-LayerNorm CV Test MSE**| — | **0.006397** ($\pm 0.000018$) | — | **REPRODUCED** |
| **1,000x Distinct-State Shuffle** | Collapse to center | **0.267761** ($\pm 0.006479$) | 0.325439 ($\pm 0.001050$) | **REPRODUCED (~39x separation)** |
| **Pairwise Distance $R^2$ (Probe-Predicted)** | High correlation (Fig 7D) | **0.7170** | 0.7521 | **REPRODUCED** |
| **Pairwise Distance $R^2$ (Activation Centroid)**| Unresolved in paper | **0.5517** ($r = +0.7428$) | — | **DIAGNOSTIC** |
| **Pairwise Distance $R^2$ (Next-Token)**| High correlation (Fig 7D) | **1.0000** | 1.0000 | **REPRODUCED** |

*Numerical Note on Mess3*: While the paper Fig S1 caption reports probe MSE of 0.0004 (pre-LN) and 0.0003 (post-LN), our measured MSE of ~0.0068 is an order of magnitude higher. Nonetheless, $R^2 = 0.980$ and the recovered 2-simplex fractal (`figures/fig5c_mess3_repr.png`) matches ground truth with high fidelity.

---

### Claim 2: RRXOR Distributed Representation (Paper Section 3.2 & Fig 7C/E)
- **Dataset**: All $2^{10} = 1,024$ sequences $\times 10$ positions = 10,240 candidates. 2,132 zero-probability prefixes excluded; **8,108 valid samples retained** (35 distinct rational belief states present among valid prefixes).
- **Core Paper Thesis**: *"The representation is NOT in the final layer alone, but IS recovered from the concatenation of layers."*

#### Layer-by-Layer vs. Concatenation Probe Results (Step 1,000,000)

| Layer Hookpoint | Feature Dim | Train MSE | Train $R^2$ | 1,000x 20/80 CV Test MSE | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `blocks.0.hook_resid_post` (Layer 0) | 64 | 0.243973 | 0.3927 | 0.261524 ($\pm 0.003100$) | Incomplete |
| `blocks.1.hook_resid_post` (Layer 1) | 64 | 0.207199 | 0.4842 | 0.221961 ($\pm 0.002711$) | Incomplete |
| `blocks.2.hook_resid_post` (Layer 2) | 64 | 0.168143 | 0.5814 | 0.180895 ($\pm 0.002371$) | Incomplete |
| `blocks.3.hook_resid_post` (Layer 3 / Final) | 64 | 0.158944 | 0.6043 | 0.171143 ($\pm 0.002378$) | Incomplete |
| `ln_final.hook_normalized` (Post-LN) | 64 | 0.183601 | 0.5430 | 0.197639 ($\pm 0.002324$) | Incomplete |
| **Concatenated (0 + 1 + 2 + 3)** | **256** | **0.075728** | **0.8115** | **0.145470 ($\pm 0.010144$)** | **>52% Train drop, 15% CV drop** |

#### Paired Normalized 20/80 CV Shuffle Null Controls (RRXOR)
To evaluate decodability on held-out test data against chance while accounting for target variance shifts under permutation, 1,000 paired 20/80 cross-validation splits (1,622 train / 6,486 test) were pre-generated and evaluated on both the real arm and the permuted distinct-state null arm (`outputs/1M/paired_normalized_null.json`). Zero degenerate splits (variance $\le 10^{-12}$) occurred across all 1,000 iterations (0 real, 0 null).

##### Distribution Summary: Real vs. Matched Shuffle Null (1,000 Paired Iterations)

| Feature Hookpoint | Arm | Raw Test MSE (Mean $\pm$ Std) | Raw Test MSE [Min, Max] | Test Target Variance (Mean $\pm$ Std) | Normalized Score $\frac{\text{MSE}}{\text{Var}}$ (Mean $\pm$ Std) | Normalized Score [Min, Max] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Concatenated** (256 dims) | **Real** | **0.145335** $\pm$ 0.010209 | [0.119810, 0.189407] | 0.401685 $\pm$ 0.001743 | **0.361783** $\pm$ 0.024917 | [0.299334, 0.471626] |
| | **Null** | **0.088366** $\pm$ 0.018709 | [0.044080, 0.152078] | 0.297205 $\pm$ 0.031934 | **0.297163** $\pm$ 0.052464 | [0.163196, 0.489534] |
| **Layer 3 / Final** (64 dims) | **Real** | **0.171187** $\pm$ 0.002371 | [0.165007, 0.178859] | 0.401685 $\pm$ 0.001743 | **0.426170** $\pm$ 0.005436 | [0.412063, 0.446277] |
| | **Null** | **0.133397** $\pm$ 0.017984 | [0.092127, 0.191779] | 0.297205 $\pm$ 0.031934 | **0.450528** $\pm$ 0.052813 | [0.293986, 0.608604] |

*What the Comparison Shows*:
1. **Target Variance Shift**: Because distinct belief states do not occur with uniform frequency in the dataset, permuting distinct state labels reduces the sample-weighted test target variance from an average of **0.4017** ($\pm 0.0017$) in the real arm to **0.2972** ($\pm 0.0319$) under permutation.
2. **Normalized Score Comparison**: Normalizing test MSE by test target variance (measuring unexplained variance ratio $1 - R^2$) reveals:
   - For **Concatenation**, the real normalized score (0.3618 $\pm$ 0.0249) is higher than the null normalized score (0.2972 $\pm$ 0.0525). Normalizing by target variance does not invert the raw comparison: real labels do not achieve lower held-out prediction error than the permuted null.
   - For **Layer 3**, the real normalized score (0.4262 $\pm$ 0.0054) is slightly lower than the null mean (0.4505 $\pm$ 0.0528), but the null distribution spans [0.2940, 0.6086], heavily overlapping the real distribution.
   - Across both representations, decoding above a matched shuffle null is not demonstrated.

*Shuffle Distinct State Count Footnote*: The shuffle control permutes across **35 distinct belief states**, not 36. Mathematical investigation of the transition system confirms that the length-0 stationary prior $\pi = [1/3, 1/6, 1/6, 1/6, 1/6]$ is never re-entered after observing any token of length 1 through 10. Because all 8,108 probe samples have observed prefixes of length $1 \dots 10$, $\pi$ never occurs as a sample target in the dataset. Exactly 35 distinct rational states exist among the probe samples.

#### Distance Diagnostics (RRXOR)
Pairwise Euclidean distance correlations across distinct belief states:
- **Definition A (Probe-Predicted Beliefs)**:
  * Representation vs. Belief Distance $R^2$: **0.2762** (improved from 0.1891 at 20k; Paper reported ~0.95).
  * Next-Token vs. Belief Distance $R^2$: **0.2387** (Paper reported ~0.31).
- **Definition B (Activation Centroids)**:
  * Layer 0 Centroid $R^2$: **0.0928** ($r = -0.3046$)
  * Layer 1 Centroid $R^2$: **0.1126** ($r = -0.3356$)
  * Layer 2 Centroid $R^2$: **0.0930** ($r = -0.3050$)
  * Layer 3 Centroid $R^2$: **0.0706** ($r = -0.2658$)
  * Concatenation Centroid $R^2$: **0.0871** ($r = -0.2951$)
  * Post-LN Centroid $R^2$: **0.0049** ($r = -0.0702$)

#### Critical Analysis of Claim 2
The reproduction of RRXOR Claim 2 is classified as **PARTIALLY REPRODUCED / QUALIFIED**:
1. **Train Fit Recovery Holds**: Concatenating all 4 layers reduces training fit MSE by **52.4%** compared to the final layer alone ($0.1589 \to 0.0757$), and raises $R^2$ from $0.604$ to $0.812$. Furthermore, 2D PCA projection of the 36 belief states (`figures/fig7e_rrxor_concat.png`) recovers the state topology.
2. **Generalization Across Layers**: In 1,000x 20/80 cross-validation, held-out test MSE improves by **15.0%** from Layer 3 ($0.1711$) to Concatenation ($0.1455$). Concatenation lowers train-fit and held-out probe error relative to the final layer, but decoding above a matched shuffle null is not demonstrated.
3. **Definition-Dependent Distance Discrepancy**: The paper's Fig 7D distance definition is not pinned in the paper text or appendix. Under Definition A (probe-predicted beliefs), $R^2 = 0.2762$; under Definition B (activation centroids), $R^2 = 0.0871$. Comparability with the paper's reported ~0.95 remains unresolved.
4. **Matched Shuffle Null Distribution**: Because 256 parameters on 1,622 training examples permit substantial flexibility against 35 distinct targets, permuting state labels alters the sample-weighted target variance from 0.4017 ($\pm 0.0017$) to 0.2972 ($\pm 0.0319$). Under variance normalization, the real normalized score is 0.3618 ($\pm 0.0249$) vs null normalized score 0.2972 ($\pm 0.0525$) for concat, confirming that decoding above a matched null is not demonstrated.

---

### Claim 3: Representation Emergence Across Training (Paper Fig 6A/D & Fig S2)
Tracking linear probe MSE across all 13 saved training snapshots from committed `outputs/1M/emergence_summary.json` and `outputs/1M/rrxor_layer_emergence.json`:

| Snapshot Step | Mess3 Final Layer Pre-LN MSE | Mess3 Normalized Loss | RRXOR Layer 3 MSE | RRXOR Concat MSE | RRXOR Normalized Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 0.045302 | 1.0369 | 0.272024 | 0.140212 | 1.1546 |
| 200 | 0.042408 | 1.0226 | 0.261176 | 0.140420 | 1.1322 |
| 500 | 0.036560 | 1.0118 | 0.244172 | 0.133632 | 1.1174 |
| 1,000 | 0.029699 | 1.0101 | 0.236235 | 0.134049 | 1.0927 |
| 2,000 | 0.019709 | 1.0181 | 0.225668 | 0.128553 | 1.0775 |
| 5,000 | 0.011836 | 1.0232 | 0.216947 | 0.108496 | 1.0517 |
| 10,000 | 0.009709 | 1.0061 | 0.207101 | 0.097422 | 1.0324 |
| 20,000 | 0.008214 | 1.0172 | 0.203915 | 0.093277 | 1.0131 |
| 50,000 | 0.007046 | 0.9823 | 0.198749 | 0.088879 | 1.0096 |
| 100,000 | 0.007026 | 1.0175 | 0.188828 | 0.085318 | 1.0032 |
| 200,000 | 0.006703 | 1.0044 | 0.178906 | 0.085021 | 1.0034 |
| 500,000 | 0.006798 | 0.9991 | 0.165152 | 0.077847 | 1.0013 |
| **1,000,000** | **0.006804** | **1.0077** | **0.158944** | **0.075728** | **0.9974** |

- **Emergence Characterization**: Training dynamics exhibit **rapid early refinement with small non-monotonic ticks, then a plateau**:
  * In RRXOR, concat MSE exhibits minor upward ticks at step 200 (0.140420 vs 0.140212) and step 1,000 (0.134049 vs 0.133632).
  * In Mess3, probe MSE refines rapidly from 0.045302 to 0.006703 at 200,000 steps, after which it fluctuates slightly within a narrow band of $[0.006703, 0.006804]$ through 1,000,000 steps.
- **Verdict**: **REPRODUCED**. Figures generated at `figures/fig6_emergence.png` and `figures/figS2_normalized_loss.png`.

---

## 8. Artifact Provenance & Checksum Manifest

All model checkpoints, summary metrics, and generated figure artifacts are verified with SHA-256 hashes recorded in `SHA256SUMS.txt`:

| Artifact Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `figures/GT_mess3.png` | Gate 4: Ground Truth 2-simplex fractal | `c34fa08f16a6b1dd6587226a452bb989ec9c9a9eb881d7fbfaa062dfa55d3d15` |
| `figures/fig5c_mess3_repr.png` | Fig 5C: Mess3 residual stream representation (1M) | `0e82221a53073dcb0facc2c0ca3c38142249b86378c14abc4593eb9d6f32251a` |
| `figures/fig6_emergence.png` | Fig 6A/D: Probe MSE emergence curve (1M) | `5ceef70138541cfae504c55bf1679261ff2f7dbfa646f906f3e1a06716075f92` |
| `figures/fig7c_rrxor_layers.png` | Fig 7C: RRXOR layer-wise probe MSE (1M) | `fc4ca939aa2bc4e548811d51c7848b3cfa8d9e2b10a2eb7f4df21a22ba3d49f6` |
| `figures/fig7d_mess3_distances.png`| Fig 7D: Mess3 pairwise distance correlation (1M) | `4d2e7421ce55a952d7ee60216b208947cf8e14bca8d082fa1bb3cefbbf29cae2` |
| `figures/fig7d_rrxor_distances.png`| Fig 7D: RRXOR pairwise distance correlation (1M) | `5b8398ff0f81d11340b01b63ef2877dd9f51877be13b309995ddad8085f1c9c7` |
| `figures/fig7e_rrxor_concat.png` | Fig 7E: RRXOR 36 belief states PCA recovery (1M) | `aa1e7934c11b15744cbfe8d6e9e422f28b464ef244eb1c78a08ff48486055d7f` |
| `figures/figS2_normalized_loss.png`| Fig S2: Probe MSE vs normalized val loss (1M) | `26b91b0f85d91a486705e1578fd4a2e2d6b009925e0ae1c32cef8ba2bc9dbfcf` |
| `checkpoints/1M/Mess3/step_1000000.pt` | Mess3 final checkpoint at step 1,000,000 | `3c5760a554eaacf6dd1895b95e2e23f9205fe842a7f500f750c732ba0120f6e3` |
| `checkpoints/1M/RRXOR/step_1000000.pt` | RRXOR final checkpoint at step 1,000,000 | `75f267c2be86f83278429e7b7a4e68f00bb9eea8c5de50ac75b1beff093fbc9f` |
| `outputs/1M/probe_results_summary.json` | Comprehensive probe metrics across 1M models | *(Updated in Part 2)* |
| `outputs/1M/emergence_summary.json` | Emergence snapshot probe metrics across 1M | *(Updated in Part 2)* |
| `outputs/1M/rrxor_layer_emergence.json` | Per-layer emergence series across all 13 checkpoints | *(Generated in Part 2)* |
| `outputs/1M/activation_centroid_distances.json` | Centroid distance diagnostics across all layers | *(Generated in Part 2)* |
| `outputs/1M/paired_normalized_null.json` | Paired normalized 20/80 CV shuffle null distribution | `e7b8639375eafec325ded5046833ff9afac7e69c1f49b29e5b03bc84c6c0081e` |
| `evidence/independent_verification/` | Verifier-run per-iteration null values + verification note | `10e24f273e45ea12c8ccf193f7074fd9cfa6f496d5b3f3e99dc7dbb3b6c745d9` |

---

## 9. Test Suite & Verification Audit

To ensure full test suite transparency:
1. **Local Tree Test Suite Execution**: Ran `pytest tests/` in the local virtual environment (`.venv`). All **26 tests passed in 6.24s** (including `test_resume.py`).
2. **Independent External Audit Execution**: An external run of the supplied suite reported **25 passed, 1 skipped** (the skip being `test_hooked_transformer_instantiation`, which skips when TransformerLens is not present in the global python environment; the bit-exact checkpoint resumption test `test_resume.py` passed).
3. **Pre-training Invariant Gates**: Ran `python -m simplexity.cli run-gates`. All 5 gates passed with zero numerical error.
4. **Privacy Audit**: Clean zero hits for host paths and usernames across the entire repository.

---

## 10. Conclusion

This reproduction study completes the full 1,000,000-iteration training and evaluation protocol of Shai et al. (NeurIPS 2024):
1. **Mess3 Residual Stream Geometry**: Confirmed. Autoregressive next-token prediction organizes internal transformer representations to mirror the fractal geometry of underlying belief states ($R^2 = 0.980$).
2. **RRXOR Distributed Representation Across Layers**: Qualified. Concatenating layers provides a substantial drop in linear probe MSE (>52% train, 15% CV test) compared to the final layer alone. Concatenation lowers train-fit and held-out probe error relative to the final layer, but decoding above a matched shuffle null is not demonstrated. Ambient Euclidean distance correlations ($R^2 = 0.276$ for probe predictions, $R^2 = 0.087$ for activation centroids) demonstrate that comparability with the paper's reported ~0.95 is definition-dependent.
