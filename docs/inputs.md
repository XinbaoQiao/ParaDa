# Input preparation

Inputs use Safetensors. S denotes source classes, C target classes, p text width,
d visual width, nm local support size, and Nq query size.

| File | Required tensor keys and shapes |
| --- | --- |
| Source | `source_text: [S,p]`, `source_visual: [S,d]` |
| Target | `target_views: [C,3,p]` |
| Each client's support | `support_features: [nm,d]`, `support_labels: [nm]` |
| Query | `features: [Nq,d]` |

Files contain exactly the listed keys. Use finite floating values and int64
support labels. Source and support feature rows must be nonzero. Empty support
has feature shape `[0,d]` and label shape `[0]`.

## Source and feature compatibility

Pair source-class text embeddings and pretrained classifier rows in exactly the
same class order. Source images and target support are not used in source training.
The CLI requires at least two aligned source classes; the old 16-neighbor
requirement does not apply to this MLP-only prior.

Source classifier rows, support features, and query features must share the
same deployed encoder feature basis. Matching dimensions is insufficient.
Keep model revisions and preprocessing fixed. The manuscript uses CLIP-B/32 text
embeddings (p=512); d depends on the vision backbone. The synthetic example uses
smaller dimensions solely for software validation.

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

## Client ownership and adaptation

All clients share an ordered target class list. Map labels to integers in `[0,C)`.
Select K examples per class in total across clients; clients may have missing
classes or no examples. The CLI checks per-class totals and requires all ten files
named `client-0.safetensors` through `client-9.safetensors` under `--clients`.
The numeric filename defines the logical client ID. The synthetic example has
three classes; real inputs need their own fixed partition and support selection.

The CLI loads these files into ten `Client` objects in one process. In every round,
each nonempty client receives the current residual, optimizes only that residual
on its own support, and returns a `Packet(client_id, count, delta)`. The server
aggregation function sees the FP32 `[C,d]` delta and sample count, with no feature
or label arguments. Actual distributed transport is not implemented. Keep packets
bound to the same episode, class order, prior and round in a distributed integration.

K=0 accepts no client support and uses the MLP prior directly. K>0 uses exactly
five rounds. See [implementation settings](../ALIGNMENT.md) for epoch budgets,
continuous learning rates, optimizer state and communication accounting.

## Evaluation and outputs

Keep support and query examples disjoint. Query features are used only for
prediction; query labels only for evaluation. Each episode starts from the same
frozen source predictor, a zero residual, and freshly seeded shuffle generators.
An adapted residual is never carried into the next episode. Sampling and metric
aggregation are external to this tensor CLI.

Adaptation writes `classifier: [C,d]`. Safetensors metadata includes method, K,
checkpoint seed, episode seed, input hashes, support count and a JSON `rounds`
record with broadcast/aggregate digests, local update counts and payload accounting.
This metadata records simulation behavior, not benchmark accuracy or network
measurements. Prediction writes `logits: [Nq,C]` and `predictions: [Nq]`, indexing
the shared class order. Outputs are write-once.
