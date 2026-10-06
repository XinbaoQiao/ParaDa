# Implementation notes

## Current method

The sole method is `parada-prototype`: one-shot client-class means/counts followed
by W0-anchored server cross-entropy fitting. W0 is formed by independently mapping
three text descriptions, normalizing each result, averaging, then normalizing.
The anchoring is the parameterization W0+Delta with Delta initially zero; there
is no additional L2 penalty or rho mixing coefficient.

| Component | Settings |
| --- | --- |
| Source MLP | Two linear layers; width 3072; GELU; dropout 0.5; output LayerNorm |
| Input masking | Probability 0.3; conditional coordinate keep probability 0.5 |
| Source training | Cosine loss; Adam; LR 0.005; batch 512; cosine decay; 500 epochs |
| Client computation | Row-normalize support features in float64; per-class mean in float64 |
| Client upload | Float32 mean plus int64 count for each active client-class row |
| Communication | One upload round; ten distinct logical clients; no local training |
| Server aggregation | Float64 count-weighted accumulation in client-ID order, cast means to float32 |
| Global prototype normalization | None; preserve the magnitude of the averaged features |
| Server objective | Count-weighted full-batch CE on scale-100 logits from row-normalized W0+Delta |
| Server optimizer | Adam, default betas/epsilon; weight decay 0; foreach/fused false |
| Server steps | T=min(75+25K,200); 100 for K=1, 200 for K=5 and K=10 |
| Server precision | Float32 prior, examples, residual, loss and optimizer; TF32 disabled in CLI |
| Prediction | Normalize W0+Delta rows; scale-100 cosine logits from normalized query features |

At step t=0 the learning rate is 1e-5. For t=1,...,T-1 it is
`0.001 * (1 + cos(pi * (t-1)/(T-2)))`, from 0.002 to zero. The server
initializes fresh Adam state once per episode and takes one full-batch update
per step. K counts examples per class globally, with exactly K reconstructed
counts for every class. Empty clients have zero active rows.

For R active client-class rows and feature width d, the upload contains `4*R*d`
mean bytes and `8*R` count bytes. Class indices, headers and prior distribution
are excluded from these tensor byte counts. They are protocol bookkeeping,
not network measurements or a formal privacy guarantee.

K=0 uses W0 without client support, uploads or optimizer steps. Query inputs are
absent from adaptation. The input preparation defaults to ten clients, Dirichlet
alpha 0.1, partition seed 3407 and global-K support selection before distribution.
Class indices refer to the same ordered episode class list. Do not replace a
frozen partition during evaluation.

## Source checkpoint reproducibility

The source initialization profile introduced in version 0.4 is retained: seed
before constructing the MLP and again before training. The initialization policy
is part of the checkpoint cache key. Version 0.3 checkpoints are not silently
reused under this profile. The loader checks source tensors, seed, file hash and
parameter-state hash. Bitwise agreement across devices or software versions is
not assumed.

The CLI enables deterministic algorithms and disables CUDA matmul/cuDNN TF32.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before initializing CUDA in applications
that call the library directly, apply the same numerical policy, and avoid
mixed-precision/autocast contexts. CUDA execution of this release is not yet
validated by the local CPU software tests.

`--episode-seed` records the episode identity; adaptation itself has no shuffling
or random optimizer operations. Input preparation selects the supports and query
with the episode seed before adaptation. Every episode starts from the same
source checkpoint, W0 and zero residual; no learned state carries over.

## Migration and release scope

Version 0.5 replaces the historical five-round residual implementation with the
current prototype method. The `Client`, `Packet` and `aggregate` residual APIs
are replaced by `PrototypePacket`, `client_packet`, `aggregate_packets` and
`fit(packets, prior, k)`. Existing source checkpoints from the version 0.4 profile
can be reused. Create new adapted outputs; old residual outputs retain their
old method identity and are not current prototype results.

The CLI reads local support files as a simulation convenience. Client packet
construction and the server packet interface are separate functions. A deployed
system must provide transport and bind packet participants, episode and class
order. Classifier metadata binds the method contract and numerical source by
SHA-256 and includes prototype and server-fit audit records.

This release covers the frozen numerical kernel and portable default fullway/
5way preparation. Historical settings also have different source families,
description corpora and full-head/native-train class pools; they are not all
replayed by the default 17-dataset input recipe. Baselines and historical result
tables are outside this package. See [reproduction status](docs/reproduction-status.json)
for input gaps and full benchmark validation limits. Software equivalence of
the adaptation kernel alone does not reproduce manuscript accuracy tables.
