# Implementation notes

## Manuscript settings

| Component | Packaged implementation | Manuscript |
| --- | --- | --- |
| Source MLP | Two linear layers; width 3072; GELU; dropout 0.5; output LayerNorm | Appendix C.1 |
| Input masking | Probability 0.3; conditional coordinate keep probability 0.5 | Appendix C.1 |
| Source training | Cosine loss; Adam; LR 0.005; batch 512; cosine decay; 500 epochs | Equation 1; Appendix C.1 |
| Text aggregation | Three independently mapped and normalized views, averaged then normalized | Equation 2 |
| Source correction | Float64 centered ridge; trace ratio 0.01; 16 neighbors; temperature 0.1 | Appendix C.2 |
| Support adaptation | Additive residual; Adam; batch 256; first-epoch LR 1e-5 then cosine decay from 0.002 | Equation 3; Appendix C.3 |
| Support budget | `min(75 + 25*K, 200)` epochs; residual coefficient 1; logit scale 100 | Appendix C.3 |

## Explicit coefficient profiles

The draft fixes the K=0 mixture coefficient at 0.5. Use `--profile manuscript`
for this setting, as shown in the README. The default `current` profile preserves
the existing implementation coefficient of 0.4. Both profiles use coefficient 1
for K>0, starting from the MLP classifier rather than the K=0 mixture.
A profile specifies numerical settings; it does not certify reproduction of
reported accuracy tables.

## Source checkpoint reproducibility

The source MLP is constructed before the training function resets the random
number generators. Thus, the requested seed alone does not determine fresh
initial weights. This package retains that ordering. Reuse the verified source
checkpoint to obtain the same trained predictor across target tasks. The loader
checks source-input identity, seed, checkpoint-file hash, and parameter-state hash.
Bitwise CPU/GPU equivalence is not assumed.

## Release coverage

The package implements standalone classifier construction and adaptation
(Sections 2.2-2.4 and Appendix C.1-C.4) on supplied tensors. Description generation
and feature extraction follow the preparation described in [inputs](docs/inputs.md).
The baseline initialization and directional regularization interfaces in
Section 2.5 require integration into each baseline's own training code and are
not exposed by this CLI. Full manuscript evaluation also requires the specified
datasets, pretrained models, descriptions, sampling, and baseline implementations.
