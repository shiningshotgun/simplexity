# DRAFT: What this taught me

I reproduced Shai et al. (NeurIPS 2024) from scratch: two tiny transformers trained for the full 1,000,000 steps each on CPU, about four hours total. For Mess3 the fractal belief-state geometry showed up in the residual stream, though the probe error stays well above the paper's caption values. The RRXOR half taught me more.

Controls have to match the claim's evaluation protocol. The paper's shuffle control permutes the belief-state labels and refits. I first ran it against training fit, and permuted labels fit better than real ones. In-sample fit is not generalization, so I reran the shuffle under the same cross-validation as the real probe. The real labels still didn't beat the null, but permuting across states also changes the target variance because states don't occur equally often, so the raw numbers are not directly comparable. The honest status is narrow: the train-fit recovery is real, and decoding above a matched null is not demonstrated.

Report numbers must come from committed artifacts. After the full runs, my summary tables quietly carried pilot values, and one column had no saved source at all. Stale or untraceable is as bad as wrong for anyone reading the repo.

Small print matters. RRXOR has 36 belief states, but one only exists before any token is observed, so the probe data has 35. The loss normalization is not the entropy rate; it's the exact finite-context optimum over the 9 scored positions. Neither changes the headline; both decide whether the repo survives a careful reader.

Resumability has to be built before the long run. My first checkpoints didn't store the sampler's RNG state, so when the first 1M attempt was interrupted, a faithful resume wasn't possible. I fixed the checkpointing, started over, and the resume machinery then carried an interrupted run to completion. The test for this runs on a toy setup: it validates the mechanism, not the 1M runs.

If I ran this again: write the null control with the claim's evaluation protocol from the start, save full RNG state before the first long run, generate tables from committed metrics, and count the states that actually occur in the data before writing "36" anywhere.
