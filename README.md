# ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning

**Anonymous implementation accompanying the manuscript.**

ParaDa treats pretrained classifier parameters as supervision for learning a
text-conditioned classifier-weight predictor. Paired source-class text embeddings
and classifier rows train a reusable MLP without revisiting source images.
For a target task, class descriptions synthesize classifier weights, and available
labeled client features refine them while the encoders and predictor stay frozen.

## Method

- **Parameter-derived supervision** (Section 2.2): learn a text-to-classifier MLP
  from pretrained source weights using cosine regression.
- **Text-side refinement** (Section 2.3): map three class descriptions independently,
  then average their normalized classifier vectors.
- **Visual-side refinement** (Section 2.4): at K=0, apply source-only ridge and
  local residual correction; at K>0, fit an additive classifier update from a
  single upload of frozen support features and labels.

Query features are used only for prediction. No source images are required.
The K>0 branch starts from the text-derived MLP classifier, independently of the
K=0 correction.

## Installation

Python 3.11 or later is required. From the repository directory:

```bash
python -m pip install -e ".[dev]"
python -m parada --help
python -m pytest -q
```

Dependencies are NumPy, PyTorch, and Safetensors. Install a PyTorch build suitable
for your device. `parada` and `python -m parada` expose the same commands.

## Quick start

The example below exercises training, zero-shot construction, one-shot adaptation,
and prediction on small synthetic tensors. It requires no dataset downloads and
is an execution example, not a benchmark evaluation.

```bash
python examples/create_demo_inputs.py

# Train the source predictor once; reuse the checkpoint across target tasks.
python -m parada train --source data/demo/source.safetensors --checkpoint outputs/demo/source --seed 42

# Zero-shot construction with the manuscript coefficient.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --k 0 --profile manuscript --output outputs/demo/k0.safetensors

# One labeled support example per target class.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --support data/demo/support.safetensors --k 1 --profile manuscript --output outputs/demo/k1.safetensors

python -m parada predict --features data/demo/query.safetensors --classifier outputs/demo/k1.safetensors --output outputs/demo/predictions.safetensors
```

Commands use CPU by default; append `--device cuda:0` to training or adaptation
for a visible logical GPU. Training uses the 500-epoch source schedule and reuses
a matching, verified checkpoint. Adaptation and prediction do not overwrite
existing outputs; use fresh output filenames for another run.

`--profile manuscript` selects the draft's K=0 coefficient of 0.5. The default
`current` profile retains 0.4; both use a K>0 residual coefficient of 1.
See [implementation notes](ALIGNMENT.md) for settings and checkpoint reproducibility.

## Using your own data

The interface accepts precomputed source text embeddings, pretrained classifier
rows, three target text views per class, and frozen support/query features.
K is the total number of selected support examples per class, across clients.
The classifier and features must use the same encoder feature basis.

[Input preparation](docs/inputs.md) specifies tensor keys, shapes, class ordering,
and the description pipeline from Appendix C.5. Prediction files contain
`logits: [N,C]` and `predictions: [N]`, indexing the supplied target-class order.

## Repository guide

| Path | Contents |
| --- | --- |
| [source.py](src/parada/source.py) | Source predictor, checkpoint handling, support adaptation |
| [correction.py](src/parada/correction.py) | Source-only ridge and residual correction |
| [pipeline.py](src/parada/pipeline.py) | Classifier construction and prediction |
| [cli.py](src/parada/cli.py) | Tensor-file command-line interface |
| [tests](tests/test_method.py) | Numerical and input-validation checks |

This package covers the standalone tensor-level method. Dataset preprocessing,
client partitioning, baseline training/integration from Section 2.5, and full
benchmark orchestration are outside the packaged interface. External datasets,
pretrained weights, and description embeddings must be prepared separately;
the tests do not reproduce the manuscript's accuracy tables.

Dependency attribution and usage terms are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
