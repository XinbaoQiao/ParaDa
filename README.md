# ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning

**Anonymous implementation accompanying the manuscript.**

ParaDa learns a text-conditioned classifier-weight predictor from source class
embeddings and pretrained classifier rows, without revisiting source images.
Target descriptions produce a classifier prior. Each client uploads its per-class
feature means and counts **once**. The server reconstructs global prototypes and
fits an additive residual against the frozen prior using cross-entropy. Encoders
and the source predictor remain frozen.

## Method

1. **Parameter-derived supervision:** train a source MLP with cosine regression.
2. **Text-side refinement:** independently map three descriptions per target
   class, normalize the predicted rows, then average and normalize to form W0.
3. **One-shot visual refinement:** clients normalize features and average them
   by class in float64. They upload float32 means and int64 counts. The server
   combines the means by count and fits a zero-initialized residual with Adam.

```text
mu_c = sum_m(n_mc * mu_mc) / sum_m(n_mc)
server logits = 100 * mu @ row_normalize(W0 + Delta).T
loss = sum_c(n_c * cross_entropy(logits_c, c)) / sum_c(n_c)
W = row_normalize(W0 + Delta)
```

Global prototypes are not normalized again. K=1 uses 100 server steps; K=5 uses
200. Clients perform no optimizer updates. K=0 uses W0 directly. See
[implementation settings](ALIGNMENT.md) and the packaged
[method contract](src/parada/method_config.json) for precision and schedule.

## Installation

Python 3.11 or later is required. From the repository directory:

```bash
python -m pip install -e ".[dev,inputs]"
python -m parada --help
python -m pytest -q
```

The numerical core needs NumPy, PyTorch, and Safetensors; the `inputs` extra adds
the image/text input dependencies used by the complete test suite. Install the
PyTorch build for your device. `parada` and `python -m parada` expose the same commands.

## Quick start

For real images and pinned pretrained models, follow the
[dataset-to-prediction reproduction guide](docs/reproduction.md).
It provides input preparation for ViT-Tiny, ConvNeXt-B and BEiT-B, explicit
dataset split contracts, description request/validation tools, and episode
sampling. Dataset images, generated descriptions, split-assignment data and
pretrained weights are not included. [Reproduction status](docs/reproduction-status.json) distinguishes
tested software from missing historical inputs and full benchmark validation.

This synthetic example requires no dataset downloads. It simulates ten logical
clients, with support sizes 2, 1, and eight zeros: globally one example per class.
It demonstrates execution, not benchmark accuracy.

```bash
python examples/create_demo_inputs.py

# Train the source predictor once (500 epochs); reuse its checkpoint.
python -m parada train --source data/demo/source.safetensors --checkpoint outputs/demo/source --seed 42

# K=0: use the text-derived classifier directly.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --k 0 --output outputs/demo/k0.safetensors

# K=1: upload class means/counts once, then perform 100 server steps.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --clients data/demo --k 1 --output outputs/demo/k1.safetensors

python -m parada predict --features data/demo/query.safetensors --classifier outputs/demo/k1.safetensors --output outputs/demo/predictions.safetensors
```

The CLI is a single-process logical-client simulation. It reads local support
files to construct client packets; `fit(packets, prior, k)` receives only class
means, counts and class indices. Network transport is outside the package. Empty
clients have no active class payload. Client and server aggregation accumulate in
float64; upload means, W0, residual and optimizer use float32, with int64 counts.
CPU is the default; `--device cuda:0` selects a visible logical GPU for training or
server fitting.

A matching source checkpoint is reused. Outputs are write-once; use fresh output
paths when repeating adaptation or prediction. For a new synthetic dataset,
run the example in a fresh working directory.

## Inputs and code

[Input preparation](docs/inputs.md) documents tensor schemas, class ordering,
descriptions, and episode boundaries. Source classifier rows, support features,
and query features must share the same encoder feature basis. Query data enter
only prediction.

| Path | Contents |
| --- | --- |
| [source.py](src/parada/source.py) | Source MLP, training, checkpoint reuse |
| [federated.py](src/parada/federated.py) | Client class means/counts, one-shot aggregation, server CE fit |
| [pipeline.py](src/parada/pipeline.py) | Input validation, global K check, episode construction |
| [cli.py](src/parada/cli.py) | Training, logical-client adaptation, prediction |
| [tests](tests) | Packet boundaries, count weighting, episode reset, input and checkpoint checks |

The core method consumes precomputed tensors; the optional `inputs` extra adds
dataset readers, model checks, feature extraction and evaluation preparation.
Raw datasets and weights are obtained separately. Baselines and full campaign
orchestration are outside this package. Software tests do not reproduce manuscript
accuracy tables. Dependency notices are
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
