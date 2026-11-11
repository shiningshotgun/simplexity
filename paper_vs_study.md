# Paper vs. Study Comparison: "Transformers Represent Belief State Geometry in their Residual Stream"

This document records the exact experimental setup from the original paper (Shai, Marzen, Teixeira, Oldenziel & Riechers, NeurIPS 2024, arXiv:2405.15943) side by side with the design, derivations, interpretations, and proposals implemented in this reproduction repository.

---

## 1. Experimental Setup Matrix

| Dimension | Original Paper (Shai et al. 2024) | This Reproduction Study (`simplexity`) | Status / Type |
| :--- | :--- | :--- | :--- |
| **Paper Source** | arXiv:2405.15943v3 / NeurIPS 2024 | Clean-room implementation from paper text and Appendix A.3–A.8 | Direct reproduction |
| **Authors' Code** | `github.com/adamimos/epsilon-transformers` (No license / All rights reserved) | Not copied or vendored; clean-room implementation | Clean-room compliance |
| **Mess3 Process** | 3 states, alphabet $\{A, B, C\}$, $T^{(x)}$ defined in App A.3 | Identical matrices $T^{(A)}, T^{(B)}, T^{(C)}$ verified against PDF p. 14 | Exact paper match |
| **RRXOR Process** | 5 states, alphabet $\{0, 1\}$, $T^{(x)}$ defined in App A.3 | Identical matrices $T^{(0)}, T^{(1)}$ verified against PDF p. 15 | Exact paper match |
| **Initial Hidden State** | Drawn from stationary distribution $\pi$ (App A.6) | Mess3: $\pi = [1/3, 1/3, 1/3]$; RRXOR: $\pi = [1/3, 1/6, 1/6, 1/6, 1/6]$ | Exact paper match |
| **Context Window** | $n_{\text{ctx}} = 10$ (App A.6) | $n_{\text{ctx}} = 10$ | Exact paper match |
| **BOS Token** | Unspecified in paper text | No BOS token used; raw token sequences of length 10 | **PROPOSAL** |
| **Target Tokens** | Next-token prediction autoregressively | Positions $0..8$ predict tokens $1..9$ (9 trained positions) | **DERIVATION** |
| **Architecture** | 4 layers, $d_{\text{model}}=64$, 1 head, $d_{\text{head}}=8$, $d_{\text{mlp}}=256$, ReLU, causal mask, LayerNorm applied (App A.6) | Exact same hyperparameters: 4 layers, $d_{\text{model}}=64$, 1 head, $d_{\text{head}}=8$, $d_{\text{mlp}}=256$, ReLU, causal mask, LayerNorm | Exact paper match |
| **Parameter Count** | Unstated in paper | Mess3: 143,075 parameters; RRXOR: 142,946 parameters | **DERIVATION** |
| **Weight Init** | Unstated (library default in TransformerLens) | PyTorch standard / Gaussian $\mathcal{N}(0, 0.02)$, LN weights 1, biases 0 | **PROPOSAL** |
| **Random Seeds** | Unstated in paper | Seed 0 everywhere | **PROPOSAL** |
| **Optimizer** | SGD, batch size 64, lr = 0.01, no weight decay (App A.6) | SGD, batch size 64, lr = 0.01, momentum = 0.0, weight decay = 0.0 | Exact paper match |
| **Training Steps** | "1,000,000 epochs" with batch 64 generated per batch (App A.6) | 1,000,000 training iterations (gradient updates) | **INTERPRETATION** |
| **Checkpoint Schedule** | Unspecified; Fig 6A shows snapshots through training | Geometric spacing: 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k, 1M | **PROPOSAL** |
| **Loss Normalization** | Fig S2 normalizes val loss such that 1.0 = optimal | $H^* = \frac{1}{9}\sum_{i=0}^8 H[X_{i+1} \mid X_{0..i}]$: Mess3 = 0.803518 nats, RRXOR = 0.586501 nats | **DERIVATION** |
| **Belief Update Rule** | $\eta' = \eta T^{(x)} / (\eta T^{(x)} \mathbf{1})$, initial $\pi$ | Exact rational (`Fraction`) for RRXOR, float64 for general analysis | Exact paper match |
| **RRXOR Reachable States** | "36 distinct belief states" (Sec 3.2, Fig 7A caption) | Exactly 36 distinct rational states (length 0 prior $\pi$ + 35 states at lengths $\ge 1$) | Exact paper match |
| **Probe Dataset** | All $3^{10} = 59,049$ seqs for Mess3; all $2^{10} = 1,024$ for RRXOR at all 10 positions | Mess3: 590,490 samples; RRXOR: 8,108 retained samples (2,132 impossible excluded) | **INTERPRETATION** |
| **Probe Regression** | OLS affine map $b = Wa + c$, uniform weighting over samples | OLS affine map $b = Wa + c$ via SVD/lstsq, uniform over valid samples | Exact paper match |
| **Residual Hookpoints** | "final layer ... before the layer norm and unembedding" (Sec 2.3) | `blocks.3.hook_resid_post` (and `blocks.{0,1,2,3}` for concat, `ln_final` for robust) | Exact paper match |
| **Controls** | Shuffle control permutes belief-state labels (Fig 6C, App A.7); 20/80 train/test CV 1000x (App A.7) | 1,000 distinct belief-state label shuffles on full dataset; 1,000 independent 20/80 CV splits on full dataset; matched 20/80 CV shuffle null | Exact paper match |
| **Distance Analysis Metric** | Pairwise Euclidean distance in representation space (Fig 7D) | Definition A (probe-predicted beliefs) and Definition B (activation centroids); comparability unresolved | **INTERPRETATION** |
| **Tooling** | TransformerLens library (App A.6) | TransformerLens installed in project-local `.venv` (`transformer-lens==2.15.4`, `torch==2.15.0`) + pure PyTorch fallback module | Exact paper match |

---

## 2. Explicit Derivations, Interpretations, and Proposals

### DERIVATION: RRXOR Stationary Distribution
- The paper states the RRXOR transition matrices $T^{(0)}$ and $T^{(1)}$ in Appendix A.3, but does not explicitly print the stationary distribution vector $\pi$.
- Solving $\pi (T^{(0)} + T^{(1)}) = \pi$ with $\sum \pi_i = 1$ yields:
  $$\pi = \left[ \frac{1}{3}, \frac{1}{6}, \frac{1}{6}, \frac{1}{6}, \frac{1}{6} \right]$$
- State 0 is visited once per 3-step cycle (probability 1/3); states 1, 2, 3, 4 are each visited with equal probability (1/6).

### DERIVATION: Optimal Loss Normalization Constant
- The paper plots "Normalized Validation Loss" in Figure S2, defining 1.0 as optimal loss, but does not provide the numerical constant or its derivation.
- In a finite context window of length 10 with no BOS, the model makes autoregressive next-token predictions at 9 positions ($i = 0, \dots, 8$). The true process conditional entropy is:
  $$H^* = \frac{1}{9} \sum_{i=0}^8 \mathbb{E}_{X_{0..i}} [H(X_{i+1} \mid X_0, \dots, X_i)]$$
  where each expectation is taken over prefixes weighted by their true process probability.
- Computed exact values (nats):
  * **Mess3**: $H^* = 0.803518$ nats
  * **RRXOR**: $H^* = 0.586501$ nats (position 0 is analytically $\ln 2 \approx 0.693147$ nats)

### INTERPRETATION: "1,000,000 Epochs"
- Appendix A.6 states: *"running for 1,000,000 epochs, and a learning rate of 0.01 ... For each batch we generated 64 sequences from the Mess3 or RRXOR HMM"*.
- In standard machine learning terminology, an "epoch" refers to a pass over a fixed finite dataset. When batches are sampled freshly from an ergodic generator at every step, "epoch" is degenerate and refers to 1,000,000 training iterations (gradient update steps).

### INTERPRETATION: Zero-Probability RRXOR Prefix Exclusion and 35 Distinct States
- In RRXOR, 2,132 of the $2^{10} \times 10 = 10,240$ candidate (sequence, position) pairs correspond to impossible prefixes under the process (where $\eta(x_{0..p}) = 0$).
- Because a belief state is undefined when conditional probability is zero, these samples are excluded from probe fitting and evaluation. 8,108 valid samples are retained.
- Furthermore, because all retained samples have observed at least one token (lengths 1 through 10) and the stationary prior $\pi = [1/3, 1/6, 1/6, 1/6, 1/6]$ is never re-entered after observing any token, exactly 35 distinct rational belief states appear in the probe dataset.

### PROPOSAL: Resolved Library Setup and Pure PyTorch Reference
- TransformerLens 2.15.4 was installed into a project-local virtual environment (`.venv`), satisfying the hard download budget (285 MB downloaded vs 5.0 GB limit).
- In addition, an exact layer-for-layer PyTorch reference architecture (`SimplexityTransformer`) was constructed and verified to produce identical parameter counts and forward/residual dynamics.

### INTERPRETATION: Pairwise Distance Analysis Representation Metrics
- In Figure 7D, the paper plots pairwise Euclidean distances in "representation space" versus true belief distances, but does not explicitly define "representation".
- We evaluate two candidate representations:
  * **Definition A**: Probe-projected belief predictions ($\hat{b} = Wa + c \in \mathbb{R}^k$), giving $R^2 = 0.717$ for Mess3 and $R^2 = 0.276$ for RRXOR.
  * **Definition B**: Activation centroids ($\bar{a}_k \in \mathbb{R}^d$, the Fig 7C "large dots"), giving $R^2 = 0.552$ for Mess3 and $R^2 = 0.087$ for RRXOR.
- Comparability with the paper's reported ~0.95 remains unresolved and is documented as a definition-dependent discrepancy.

### METHODOLOGY NOTE: Matched 20/80 CV Shuffle Null
- To rigorously test held-out decodability above chance, we evaluate belief-state label permutations under the identical 1,000-iteration 20/80 cross-validation protocol used for real models.
- For RRXOR, the matched CV shuffle null yields held-out test MSE of 0.0875 ($\pm 0.0179$) for concat and 0.1327 ($\pm 0.0169$) for Layer 3, reflecting altered target variance under permutation.
