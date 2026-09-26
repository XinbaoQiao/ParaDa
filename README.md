# ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning

**Anonymous implementation accompanying the manuscript.**

ParaDa learns a text-conditioned classifier-weight predictor from source class
embeddings and pretrained classifier rows, without revisiting source images.
Target descriptions produce a classifier prior. Client support examples refine
this prior through **five rounds of federated residual adaptation**, with the
encoders and source predictor frozen.

## Method

1. **Parameter-derived supervision:** train a source MLP with cosine regression.
2. **Text-side refinement:** independently map three descriptions per target
   class, normalize the predicted rows, then average and normalize to form W0.
3. **Visual refinement:** initialize the residual to zero. In each of five rounds,
   active clients start from the broadcast residual and optimize local cross-entropy.
   The server averages their raw residuals using support sample counts.

```text
local logits = 100 * row_normalize(Xm) @ row_normalize(W0 + Delta_m).T
Delta = sum_m (nm / sum_j nj) * Delta_m
W = row_normalize(W0 + Delta)
```

Aggregation does not normalize the residual or average normalized classifier rows.
K=1 uses 20 local epochs per round; K=5 uses 40. Adam state resets each round,
while the learning-rate schedule continues across rounds. K=0 uses W0 directly.
See [implementation settings](ALIGNMENT.md) for the complete schedule.

## Installation

Python 3.11 or later is required. From the repository directory:

```bash
python -m pip install -e ".[dev]"
python -m parada --help
python -m pytest -q
```

Dependencies are NumPy, PyTorch, and Safetensors. Install the PyTorch build for
your device. `parada` and `python -m parada` expose the same commands.

## Quick start

This synthetic example requires no dataset downloads. It simulates ten logical
clients, with support sizes 2, 1, and eight zeros: globally one example per class.
It demonstrates execution, not benchmark accuracy.

```bash
python examples/create_demo_inputs.py

# Train the source predictor once (500 epochs); reuse its checkpoint.
python -m parada train --source data/demo/source.safetensors --checkpoint outputs/demo/source --seed 42

# K=0: use the text-derived classifier directly.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --k 0 --output outputs/demo/k0.safetensors

# K=1: run five rounds over client-0.safetensors through client-9.safetensors.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --clients data/demo --k 1 --output outputs/demo/k1.safetensors

python -m parada predict --features data/demo/query.safetensors --classifier outputs/demo/k1.safetensors --output outputs/demo/predictions.safetensors
```

The CLI is a single-process logical-client simulation. Each `Client` owns its
support tensors; the aggregation API receives only residual/count packets.
Network transport is outside the package. Empty clients send no delta payload.
Training and residual updates use FP32. CPU is the default; `--device cuda:0`
selects a visible logical GPU for training or adaptation.

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
| [federated.py](src/parada/federated.py) | Local CE updates, five-round schedule, residual aggregation |
| [pipeline.py](src/parada/pipeline.py) | Input validation, global K check, episode construction |
| [cli.py](src/parada/cli.py) | Training, logical-client adaptation, prediction |
| [tests](tests) | Round state, sample weighting, episode reset, input and checkpoint checks |

The package implements the standalone method on precomputed tensors. Datasets,
feature extraction, baseline integrations, and benchmark orchestration are external.
Software tests do not reproduce manuscript accuracy tables. Dependency notices are
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
