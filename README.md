# ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning

ParaDa learns a text-to-classifier MLP from paired source-class text embeddings
and pretrained classifier rows. Three descriptions specify each target class.
At K=0, a source-only ridge and local residual correction complements the MLP.
At K>0, the server fits an additive classifier update from frozen support features
uploaded once by clients. Visual and text encoders remain frozen.

This is a focused, runnable export of the current numerical implementation.
It includes source training with checkpoint reuse, target classifier construction,
support adaptation, prediction, and correctness tests. It operates on precomputed
embeddings and features. It does not include datasets, pretrained weights, generated
descriptions, result tables, or a full raw-image benchmark runner. The tests validate
software behavior; they do not reproduce the manuscript's accuracy tables.

**Read [ALIGNMENT.md](ALIGNMENT.md) before claiming paper reproduction.** The current
K=0 coefficient is 0.4; the supplied manuscript specifies 0.5. The explicit
`--profile manuscript` switch selects 0.5. Both profiles use K>0 coefficient 1.
The default `current` profile preserves the current execution setting.

## Installation

Python 3.11 or later:

```bash
python -m pip install -e '.[dev]'
parada --help
pytest -q
```

`python -m parada` provides the same commands. Install the appropriate PyTorch
build for your device. All examples run on CPU; use `--device cuda:0` for the logical
GPU selected by your environment. No dependency download or training starts merely
by importing the package.

## Inputs

Use `safetensors.torch.save_file` to write the following files. All floating inputs
must be finite. Source rows must be nonzero and aligned by source class ID. Target
rows, support labels, and output classifier rows share one class order.

| File | Tensor keys and shapes | Meaning |
| --- | --- | --- |
| `source.safetensors` | `source_text: [S,p]`, `source_visual: [S,d]` | Frozen source text embeddings and corresponding pretrained classifier rows |
| `target.safetensors` | `target_views: [C,3,p]` | Appearance, functionality, and environment descriptions for each target class |
| `support.safetensors` | `support_features: [C*K,d]`, `support_labels: [C*K]` | Selected frozen client features and dense int64 labels in `[0,C)` |
| `query.safetensors` | `features: [N,d]` | Frozen query features, used only for prediction |

The manuscript uses CLIP-B/32 text width p=512. Visual width d follows the chosen
backbone. The correction requires at least 16 distinct normalized source-text rows.
Source image data are never required by this package. For raw-image reproduction,
prepare features with the exact pretrained backbone and preprocessing, and retain
the class-order, model-revision, split, seed, and input hashes externally.

The manuscript's descriptions are three short GPT-4o outputs per class emphasizing
appearance, functionality, and environment. Only the class name goes to the generator;
no client images or query labels are inputs. Encode each description separately with
the frozen text encoder. Text generation is an external preparation step: this export
does not regenerate, distribute, or assert the identity of historical descriptions.

## Train once, reuse across tasks

```bash
parada train --source data/source.safetensors --checkpoint outputs/source-seed42 --seed 42
```

Source training uses hidden width 3072, GELU, hidden dropout 0.5, LayerNorm, the
existing input masking rule, Adam at 0.005, batch 512, and cosine decay for 500 epochs.
Source targets are normalized. The checkpoint includes source-input and parameter
hashes; rerunning with the same inputs reuses it. A partial or incompatible checkpoint
is rejected. See the initialization limitation in [ALIGNMENT.md](ALIGNMENT.md).

## Construct and adapt

```bash
# Current implementation profile, no target support.
parada adapt --source data/source.safetensors --checkpoint outputs/source-seed42 \
  --target data/target.safetensors --k 0 --output outputs/k0.safetensors

# Explicit manuscript coefficient; this is not evidence of reproduced accuracy.
parada adapt --source data/source.safetensors --checkpoint outputs/source-seed42 \
  --target data/target.safetensors --k 0 --profile manuscript --output outputs/k0-paper.safetensors

# One labeled support example per class, pooled from the clients' single upload.
parada adapt --source data/source.safetensors --checkpoint outputs/source-seed42 \
  --target data/target.safetensors --support data/support.safetensors \
  --k 1 --output outputs/k1.safetensors

parada predict --features data/query.safetensors --classifier outputs/k1.safetensors \
  --output outputs/query-predictions.safetensors
```

K>0 starts from the MLP classifier, not the K=0 mixture. Only the additive residual is
trainable. Adaptation uses Adam, batch 256, learning rate 1e-5 for the first epoch,
then cosine decay from 0.002 to zero, for `min(75 + 25*K, 200)` epochs. Cosine logits
use scale 100. Query labels never enter fitting. Existing output files are rejected.

For episodic use, select the episode's class rows, remap support labels to that order,
and provide exactly K support examples per selected class. Client partitioning and
support/query sampling remain the responsibility of the experiment preparation.
The manuscript's main protocol uses 10 clients, Dirichlet alpha 0.1, partition seed
3407, and evaluation seeds 42 through 46. These dataset protocols and the reported
baseline training runs are not recreated by the tensor-level commands above.

## Source layout

- `src/parada/source.py`: existing source MLP, checkpoint handling, and residual optimizer.
- `src/parada/correction.py`: existing source dictionary, stable neighbors, and ridge correction.
- `src/parada/pipeline.py`: composition with explicit coefficients and input checks.
- `src/parada/cli.py`: portable tensor-file interface.
- `tests/test_method.py`: mathematical and input-boundary checks.

The federation exchanges features and labels, not raw images. This protocol does not
provide differential privacy or confidentiality of the uploaded representations.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependencies and attribution.
No project-wide redistribution license is granted in this review snapshot.
