# Input preparation

The commands operate on Safetensors files written with
`safetensors.torch.save_file`. S denotes source classes, C target classes, p text
width, d visual feature width, and N query examples.

| File | Required tensor keys and shapes |
| --- | --- |
| Source | `source_text: [S,p]`, `source_visual: [S,d]` |
| Target | `target_views: [C,3,p]` |
| Support (K>0 only) | `support_features: [C*K,d]`, `support_labels: [C*K]` |
| Query | `features: [N,d]` |

Each file must contain exactly these keys. Use finite floating-point values and
int64 support labels. The manuscript's transmitted features and weights are FP32.

## Source and feature compatibility

Pair each source-class text embedding with the corresponding pretrained classifier
row, preserving class identifiers and row order. Source rows must be nonzero.
The correction requires at least 16 distinct normalized source-text rows.
Source images and target support do not enter source-predictor training.

Use the same visual encoder feature basis for source classifier rows, client
support features, and query features. Matching their dimension alone is
insufficient. Keep the pretrained model revision and preprocessing fixed.
The manuscript uses CLIP-B/32 text embeddings (p=512); visual width d depends on
the backbone. The synthetic example uses smaller dimensions only to exercise
this interface.

## Target descriptions (Appendix C.5)

Generate three descriptions from each target class name alone:
`appearance`, `functionality`, and `environment`. The manuscript uses GPT-4o.
Each description contains 8-18 English words and differs from the other two.
Validate exact class-name spelling and input order, with one complete record per
class. Client images, support examples, queries, and query labels are not inputs
to the generator. Description acceptance must not depend on target accuracy.

Insert each accepted description into this text-encoder template:

```text
A photo of a <class_name>: <description>
```

Check the CLIP text encoder's 77-token context limit and encode each prompt
separately with the frozen encoder. Stack the three embeddings in appearance,
functionality, environment order to obtain `[C,3,p]`. Do not concatenate the
three descriptions or average their embeddings before the MLP. The predictor
maps each view separately, normalizes its output, then averages and normalizes
in visual space (Equation 2).

Generation and encoding are external preparation steps. Reuse the same accepted
descriptions and embeddings when comparing configurations.

## Support and query separation

At K=0, omit the support file. At K>0, provide exactly K examples for every
selected class, pooled across clients after frozen feature extraction. Labels
must be dense integers in `[0,C)` and follow the target tensor's class order.
The server optimizes only an additive classifier update; encoders and source
predictor remain fixed. Features and labels are disclosed to the server, so this
interface does not provide a formal privacy guarantee.

For 5-way evaluation, select five class rows and remap labels accordingly. Keep
support and query images disjoint. Every episode starts from the same frozen
source state, with no adapted-state carryover. The manuscript uses 10 clients,
Dirichlet alpha 0.1, partition seed 3407, seeds 42-46, and 20 independent 5-way
trials per seed. Client sampling and evaluation aggregation are external to the CLI.

## Outputs and reuse

Adaptation writes `classifier: [C,d]` with profile, K, seed, and input/checkpoint
hash metadata. Prediction writes cosine `logits: [N,C]` and class indices
`predictions: [N]`. Keep the class-order mapping with these files.

A source checkpoint can be reused only with matching source tensors and seed.
To repeat adaptation or prediction, choose new output filenames. The demo input
script likewise refuses to replace existing data. See
[implementation notes](../ALIGNMENT.md) for initialization and profile details.
