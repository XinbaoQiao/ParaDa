# Implementation notes

## Current method

The source MLP produces a frozen, three-view classifier prior W0. For K>0, ten
logical clients refine an additive residual over five communication rounds.
Every episode initializes Delta to zero and resets client shuffle generators.

| Component | Settings |
| --- | --- |
| Source MLP | Two linear layers; width 3072; GELU; dropout 0.5; output LayerNorm |
| Input masking | Probability 0.3; conditional coordinate keep probability 0.5 |
| Source training | Cosine loss; Adam; LR 0.005; batch 512; cosine decay; 500 epochs |
| Text aggregation | Map three views, normalize each output, average, normalize |
| Local objective | Cross-entropy on scale-100 cosine logits from W0+Delta; frozen W0 |
| Local optimizer | Adam, default betas/epsilon, weight decay 0; batch 256; fresh moments each round |
| Local epoch budget | Total E=min(75+25K,200); E/5 epochs in each of five rounds |
| Learning rate | First global epoch 0.00001, then cosine decay from 0.002 to 0 |
| Client shuffle | CPU generator seeded with episode_seed + 1000003 * client_id |
| Server aggregation | Sample-count-weighted mean of raw residual matrices in client-ID order |
| Precision | FP32 features, prior, residual, optimizer and aggregation |
| Prediction | Normalize W0+Delta rows; scale-100 cosine logits |

For zero-based global epoch t, the rate is 1e-5 at t=0. For t=1,...,E-1 it is
`0.001 * (1 + cos(pi * (t-1)/(E-2)))`. This schedule continues across rounds;
it does not restart when Adam moments reset. K=1 gives 20 local epochs per round;
K=5 and K=10 give 40. K counts examples per class globally across all clients.

At each round, all nonempty clients start from the same broadcast Delta. Each
uploads its full updated residual, not a normalized classifier or an increment
relative to the broadcast. The server computes `sum(nm * Delta_m) / sum(nm)`.
Empty clients receive/send no residual payload and take no optimizer steps.
For C classes and width d, one active client's residual payload is `4*C*d` bytes
in each direction per round, plus an 8-byte uploaded count; headers and initial
prior distribution are excluded. Counts are Python integers in this simulator;
the byte accounting assumes an int64 transport representation.

K=0 uses W0 without residual training. Query inputs are absent from adaptation.
The CLI validates all ten support files and exactly K examples per class. Dataset
partitioning is external; the manuscript uses ten clients, Dirichlet alpha 0.1,
and partition seed 3407. Do not replace a frozen partition during evaluation.

## Source checkpoint reproducibility

Source training is unchanged. The MLP is constructed before the training function
resets random generators, so the requested seed alone does not determine fresh
initial weights. Reuse a verified source checkpoint across target tasks. The
loader checks source tensors, seed, file hash, and parameter-state hash.
`--episode-seed` controls support shuffling separately from the checkpoint seed;
it defaults to `--seed`. For repeated episodes, supply the protocol's episode
seed (the evaluation schedule uses `seed + 10007 * episode_index`, zero-based).
Bitwise agreement across devices or software versions is not assumed.

## Migration and release scope

Version 0.3 replaces the one-shot ridge path with five-round local cross-entropy
updates. The old `client-stats`, `--statistics`, and `--regularization` options
are removed. Pass a directory of ten support files with `--clients` for K>0.
Compatible source checkpoints can still be reused; create new adapted outputs.

The CLI simulates client ownership in one process. A deployed system must provide
transport and enforce episode, class-axis and participant identity around the
`Client`, `Packet`, and `aggregate` interfaces. Delta/count exchange alone makes
no formal privacy guarantee. Baseline integrations and full data preparation are
outside this package; this release does not introduce an accuracy claim.
