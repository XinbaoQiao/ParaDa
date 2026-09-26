# Input preparation

All inputs use Safetensors. S denotes source classes, C target classes, p text
width, d visual feature width, nm local support size, and Nq query size.

| File | Required tensor keys and shapes | Location |
| --- | --- | --- |
| Source | `source_text: [S,p]`, `source_visual: [S,d]` | Source training / server |
| Target | `target_views: [C,3,p]` | Classifier construction |
| Local support | `support_features: [nm,d]`, `support_labels: [nm]` | Client only |
| Uploaded packet | `gram_upper: [d*(d+1)/2]`, `cross: [d,C]`, `count: []` | Server |
| Query | `features: [Nq,d]` | Prediction only |

Files contain exactly the listed keys. Use finite floating values and int64
labels/counts. Source and support feature rows must be nonzero. Empty local
support has shape `[0,d]` and labels `[0]`. Client counts need not match.
`client-stats` supplies packet format and client ID metadata automatically;
IDs must be unique within the task. The upper triangle follows
`torch.triu_indices(d,d)` order, including the diagonal.

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

## Client statistics and server solve

All clients share the same ordered target class list. Map labels to integers
in `[0,C)` before accumulation. For a K-shot task, select K examples per class
in total across clients. Individual clients may have missing classes or no samples.
This class-order and selection contract is established outside the packet format.

Run `client-stats` locally for each client's support file. It normalizes features,
accumulates `Gm = Xm.T @ Xm` and `Hm = Xm.T @ Ym` in FP64 (without materializing
dense one-hot Ym), packs Gm's upper triangle and transmits FP32 statistics plus
an int64 count. Hm contains class-wise feature sums, including zero columns for
missing classes. Raw features and per-example labels stay client-local.
`--communication-dtype float64` is available for numerical checks.

For K>0, send exactly one packet per client to server `adapt --statistics ...`.
Clients do not need the classifier prior to construct their packets. The server
sums Gm, Hm and nm, solves the residual system with regularization 0.01, then
normalizes and returns the classifier. No local epochs, support learning rate,
or iterative aggregation rounds are used. Packets must belong to the same task,
class axis and feature basis; do not combine packets from separate episodes.

K=0 accepts no statistics and uses the MLP prior directly. Installed source
artifacts permit local zero-shot construction without target-task communication.
These statistics do not guarantee privacy: a class sum can reveal an individual
feature when that class has only one local example.

## Evaluation and outputs

Keep support and query examples disjoint. Query features are used only for
prediction; query labels only for evaluation. For each episode, select its classes,
remap labels, create fresh packets, and start from the same frozen source predictor.
No adapted-state carryover is allowed. Dataset sampling and aggregation are external
to the tensor CLI; follow the experiment's specified protocol.

Adaptation writes `classifier: [C,d]` with method, K, seed, regularization, support
count, and input/checkpoint hashes. Prediction writes `logits: [Nq,C]` and
`predictions: [Nq]`, indexing the shared class order. Checkpoints require matching
source tensors and seed; outputs are write-once. See
[implementation notes](../ALIGNMENT.md) for reproducibility and migration details.
