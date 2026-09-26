# Implementation notes

## Current method

K=0 uses only the three-view MLP classifier W0 (Equation 2). K>0 solves the
prior-centered ridge objective from client sufficient statistics (Section 2.4).
There is no source-neighbor correction, K=0 mixture coefficient, or support-side
optimizer. The default ridge regularization is 0.01, independent of query labels.

| Component | Settings |
| --- | --- |
| Source MLP | Two linear layers; width 3072; GELU; dropout 0.5; output LayerNorm |
| Input masking | Probability 0.3; conditional coordinate keep probability 0.5 |
| Source training | Cosine loss; Adam; LR 0.005; batch 512; cosine decay; 500 epochs |
| Text aggregation | Three independently mapped and normalized views, averaged then normalized |
| Client statistics | Normalize frozen features, accumulate XTX and XTY in FP64 |
| Transmission | Packed symmetric upper triangle, dense class sums, int64 count; default FP32 statistics |
| Server | Sum in FP64; solve `(G + N*lambda*I) Delta^T = H - G W0^T` with `torch.linalg.solve` |
| Prediction | Normalize W0+Delta rows after solving; FP32 classifier; scale-100 cosine logits |

The client API does not require W0. The server API receives only statistic
packets. Counts are summed, not used to average per-client means. Empty clients
contribute dense zeros and count zero; all-empty support is rejected for K>0.
Duplicate client IDs, malformed packets, and mixed transmission dtypes are rejected.
The CLI checks the global total N=C*K. Class order and per-class K selection must
be established during client preparation; these cannot be certified from the
transmitted sums and total count alone.

Tests compare the result with an independently constructed augmented least-squares
problem, including unequal client sizes and empty clients. FP64 communication is
available for equivalence checks; the manuscript uses FP32 transmission. The
statistic payload size is `bytes_per_float * (d*(d+1)/2 + d*C) + 8` per client,
excluding file headers and routing metadata. One upload is not a guarantee of
fewer bytes than alternative payloads.

## Source checkpoint reproducibility

Source training is unchanged. The MLP is constructed before the training function
resets the random generators, so the requested seed alone does not determine fresh
initial weights. Reuse a verified source checkpoint across target tasks. The loader
checks source tensors, seed, file hash, and parameter-state hash. Bitwise CPU/GPU
equivalence is not assumed.

## Migration and release scope

Version 0.2 replaces support-feature upload and iterative cross-entropy adaptation
with `client-stats` and server `--statistics` inputs. The former `--support` and
`--profile` adaptation options are removed. K=0 is now MLP-only; old mixed
classifiers and adapted outputs must not be interpreted as outputs of this method.
Compatible source checkpoints can still be reused. Use new output paths.

The package implements the standalone method on precomputed tensors. Baseline
initialization/regularization interfaces from Section 2.5 require integration
with the respective baseline training code and remain outside this CLI. Real-data
results require the specified inputs and evaluation protocol in addition to the
method implementation. No new accuracy claim is implied by this software release.
