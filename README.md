# ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning

**Anonymous code submission for peer review.**

ParaDa uses a pretrained vision classifier as supervision for a text-to-classifier
mapper. It transfers to target classes through text descriptions, with optional
adaptation from a single upload of labeled client features. Both visual and text
encoders remain frozen.

This repository provides the numerical method and a portable tensor interface:
source training, checkpoint reuse, zero-shot classifier construction, few-shot
adaptation, and prediction.

## Method overview

1. **Learn from pretrained parameters.** Fit an MLP from source-class text
   embeddings to the corresponding pretrained visual classifier rows. Source
   images are not required.
2. **Describe target classes.** Encode three descriptions per class, covering
   appearance, functionality, and environment. Map each view independently,
   normalize its classifier row, then average and normalize across views.
3. **Construct the target classifier.** At K=0, combine the MLP with a source-only
   ridge and local residual correction. At K>0, fit an additive update to the MLP
   classifier using selected labeled support features.
4. **Predict with frozen features.** Score queries using cosine logits with
   scale 100. Query features and labels do not enter adaptation.

| Setting | Target support | Classifier |
| --- | --- | --- |
| Zero-shot, K=0 | None | Normalized mixture of MLP and source residual classifiers |
| Few-shot, K=1 through 10 | Exactly K examples per target class | MLP classifier plus a support-trained residual |

The few-shot branch starts from the MLP classifier, independently of the zero-shot
mixture. K is the total support count per class supplied to the server interface;
client partitioning is an external preparation step.

## Installation

Use Python 3.11 or later. From the repository directory:

```bash
python -m pip install -e ".[dev]"
python -m parada --help
python -m pytest -q
```

Runtime dependencies are NumPy, PyTorch, and Safetensors; version ranges are in
[pyproject.toml](pyproject.toml). Install a PyTorch build suitable for your device.
The `parada` command is equivalent to `python -m parada`. Examples below use CPU;
append `--device cuda:0` to training or adaptation for a visible logical GPU.

## Quick start

This complete synthetic example needs no datasets, pretrained weights, or generated
descriptions. Random tensors exercise the interface; predictions are not benchmark
results. The training command uses the default 500-epoch schedule.

```bash
python examples/create_demo_inputs.py

# Train once; repeating this command reuses the verified source checkpoint.
python -m parada train --source data/demo/source.safetensors --checkpoint outputs/demo/source --seed 42

# Zero-shot classifier.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --k 0 --output outputs/demo/k0.safetensors

# Adapt with one labeled example per class.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --support data/demo/support.safetensors --k 1 --output outputs/demo/k1.safetensors

# Predict on held-out feature rows.
python -m parada predict --features data/demo/query.safetensors --classifier outputs/demo/k1.safetensors --output outputs/demo/predictions.safetensors
```

Prediction files contain `logits: [N,C]` and `predictions: [N]`. Predictions index
rows in the supplied target-class order. Adaptation files contain
`classifier: [C,d]` and metadata recording the profile, K, seed, checkpoint hash,
and input hashes.

Training reuses a complete checkpoint only when its source inputs and seed match;
incomplete or incompatible checkpoints are rejected. Adaptation and prediction
refuse to overwrite existing outputs: choose a new output filename when repeating
these commands. The demo-input script also refuses to replace existing files.

## Preparing real inputs

Use `safetensors.torch.save_file`. Floating inputs must be finite, and support
labels must be `torch.int64`. S denotes source classes, C target classes, p text
embedding width, d visual feature width, and N query examples.

| File | Required tensor keys and shapes | Contents |
| --- | --- | --- |
| `source.safetensors` | `source_text: [S,p]`, `source_visual: [S,d]` | Aligned source text embeddings and pretrained visual classifier rows |
| `target.safetensors` | `target_views: [C,3,p]` | Three text embedding views per target class |
| `support.safetensors` | `support_features: [C*K,d]`, `support_labels: [C*K]` | Selected client features and labels in `[0,C)`; required for K>0 |
| `query.safetensors` | `features: [N,d]` | Frozen query features for prediction |

Each file must contain exactly the listed keys. Source rows must be nonzero and
aligned by source class ID. The source correction requires at least 16 distinct
normalized source-text rows. Target views, support labels, and output classifier
rows must share the same class order, with exactly K support examples per class.

The manuscript uses CLIP-B/32 text embeddings (p=512); d follows the visual
backbone. Prepare image features using the matching pretrained weights and
preprocessing. Keep model revisions, descriptions, class order, data splits,
support selection, seeds, and input hashes fixed for comparisons.

The manuscript specifies three short GPT-4o descriptions per class, generated
from its name alone, covering appearance, functionality, and environment. Encode
each description separately. Description generation and feature extraction are
external preparation steps; historical assets are not supplied in this package.

For episodic evaluation, select the episode's classes, remap support labels to
that order, and start every episode from the same frozen source checkpoint. Keep
support and query samples disjoint, without carrying adapted state across episodes.
The manuscript's main protocol specifies 10 clients, Dirichlet alpha 0.1, partition
seed 3407, and evaluation seeds 42 through 46. The tensor interface does not
implement that sampling or recreate baseline training runs.

## Training settings and implementation profiles

| Component | Settings |
| --- | --- |
| Source mapper | Hidden width 3072, GELU, dropout 0.5, LayerNorm, input masking |
| Source optimization | Adam, learning rate 0.005, batch size 512, cosine decay, 500 epochs |
| Source correction | Centered ridge with trace-scaled regularization 0.01; 16 neighbors; temperature 0.1 |
| Support optimization | Additive residual only; Adam; batch size 256; `min(75 + 25*K, 200)` epochs |
| Support learning rate | First epoch at 1e-5, then cosine decay from 0.002 to zero |
| Prediction | Normalized feature/classifier rows; cosine logit scale 100 |

The manuscript coefficient is available through an explicit profile:

| `--profile` | K=0 mixture coefficient | K>0 residual coefficient |
| --- | --- | --- |
| `current` (default) | 0.4 | 1.0 |
| `manuscript` | 0.5 | 1.0 |

Append `--profile manuscript` to an `adapt` command to select 0.5 at K=0. The default
preserves the existing execution setting. A profile selects coefficients; it does
not establish reproduction of the manuscript's accuracy tables.

[ALIGNMENT.md](ALIGNMENT.md) explains coefficient selection and an initialization
limitation: fresh MLP weights are created before the training function resets the
random generators. The requested seed alone therefore does not identify fresh
initial weights. Reuse a verified source checkpoint to preserve the same trained
state. Bitwise CPU/GPU equivalence is not claimed.

## Code guide and validation

| File | Responsibility |
| --- | --- |
| [source.py](src/parada/source.py) | Source MLP, training, checkpoint validation, and support optimizer |
| [correction.py](src/parada/correction.py) | Source dictionary, stable neighbors, ridge/residual correction |
| [pipeline.py](src/parada/pipeline.py) | Method composition, input checks, and prediction |
| [cli.py](src/parada/cli.py) | Tensor-file command-line interface |
| [test_method.py](tests/test_method.py) | Mathematical checks, checkpoint integrity, and input-boundary tests |
| [create_demo_inputs.py](examples/create_demo_inputs.py) | Synthetic inputs for the quick start |

Run `python -m pytest -q` to check three-view aggregation, ridge/residual correction,
profile behavior, support adaptation, checkpoint integrity, and malformed inputs.

This submission includes method code and correctness tests. Datasets, pretrained
weights, historical description embeddings, result tables, and a raw-image
benchmark driver are not bundled. Full table reproduction requires those matched
inputs and evaluation protocols in addition to this implementation.

## Dependencies and use

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and dependency
notices. External inputs remain subject to their respective terms. This review
snapshot grants no final project-wide redistribution license.
The method uploads features and labels rather than raw images; it does not provide
differential privacy or confidentiality of the uploaded representations.
