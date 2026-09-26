# ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning

**Anonymous implementation accompanying the manuscript.**

ParaDa learns a text-conditioned classifier-weight predictor from paired source
class embeddings and pretrained classifier rows, without revisiting source images.
Target descriptions produce a classifier prior. Labeled client examples refine it
through a one-shot analytic residual solve, with the encoders and predictor frozen.

## Method

1. **Parameter-derived supervision:** train a source MLP with cosine regression.
2. **Text-side refinement:** independently map three descriptions per target
   class, normalize the predicted rows, then average and normalize to form W0.
3. **Analytic visual refinement:** each client uploads sufficient statistics
   once; the server solves a ridge residual centered at W0.

For K=0, the classifier is W0 directly. For K>0, with normalized client features
Xm and one-hot labels Ym, the objective is

```text
min_Delta  (1/N) sum_m ||Xm (W0 + Delta)^T - Ym||_F^2 + lambda ||Delta||_F^2

G = sum_m Xm^T Xm,    H = sum_m Xm^T Ym,    N = sum_m nm
(G + N lambda I) Delta^T = H - G W0^T
W = row_normalize(W0 + Delta)
```

The manuscript uses `lambda = 0.01`. Client accumulation and the server solve
use FP64; statistics are transmitted in FP32 by default. Clients send the upper
triangle of Gm, the full Hm, and an int64 count, including zero packets for empty
clients. Summing counts and statistics preserves sample-mean weighting for unequal
client sizes. No support features or per-example labels enter the server interface.

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

This synthetic example requires no dataset downloads. It exercises three clients
with unequal support sizes (2, 1, and 0) and a total of one example per target
class. It demonstrates execution, not benchmark accuracy.

```bash
python examples/create_demo_inputs.py

# Train the source predictor once (500 epochs); reuse its checkpoint.
python -m parada train --source data/demo/source.safetensors --checkpoint outputs/demo/source --seed 42

# K=0: use the text-derived MLP classifier directly.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --k 0 --output outputs/demo/k0.safetensors

# Run each command on its respective client. Only the output packet is uploaded.
python -m parada client-stats --support data/demo/client-0.safetensors --classes 3 --client-id 0 --output outputs/demo/client-0-stats.safetensors
python -m parada client-stats --support data/demo/client-1.safetensors --classes 3 --client-id 1 --output outputs/demo/client-1-stats.safetensors
python -m parada client-stats --support data/demo/client-2.safetensors --classes 3 --client-id 2 --output outputs/demo/client-2-stats.safetensors

# Server: aggregate packets and solve the analytic residual with lambda=0.01.
python -m parada adapt --source data/demo/source.safetensors --checkpoint outputs/demo/source --target data/demo/target.safetensors --statistics outputs/demo/client-0-stats.safetensors outputs/demo/client-1-stats.safetensors outputs/demo/client-2-stats.safetensors --k 1 --output outputs/demo/k1.safetensors

python -m parada predict --features data/demo/query.safetensors --classifier outputs/demo/k1.safetensors --output outputs/demo/predictions.safetensors
```

Commands use CPU by default; `--device cuda:0` selects a visible logical GPU for
source training or MLP inference. The analytic solve runs in FP64 on CPU.
A matching source checkpoint is reused. Outputs are write-once; choose fresh
filenames when repeating adaptation or prediction. For another example run,
use a fresh working directory. Do not mix packets from different tasks or episodes.

## Inputs and code

[Input preparation](docs/inputs.md) documents tensor schemas, class ordering,
description generation, and client/server boundaries. The visual features and
classifier prior must use the same encoder feature basis, not merely the same
width. Query data are used only after classifier construction.

| Path | Contents |
| --- | --- |
| [source.py](src/parada/source.py) | Source MLP, training, and checkpoint reuse |
| [sufficient_stats.py](src/parada/sufficient_stats.py) | Client statistics and analytic residual solver |
| [packet_io.py](src/parada/packet_io.py) | Portable statistic packets |
| [pipeline.py](src/parada/pipeline.py) | MLP-only K=0 and ridge K>0 classifier construction |
| [cli.py](src/parada/cli.py) | Client statistics, adaptation, and prediction commands |
| [tests](tests/test_method.py) | Independent least-squares parity, packet and checkpoint checks |

Exact statistics recover the centralized solution of this quadratic objective.
FP32 transmission introduces rounding; equivalence is numerical, not bitwise.
The final row normalization occurs after the solve. This objective is distinct
from iterative cross-entropy adaptation. Statistics alone do not guarantee privacy.

[Implementation notes](ALIGNMENT.md) give settings, checkpoint reproducibility,
and migration from the previous interface. The package covers the standalone
tensor-level method; datasets, feature extraction, baseline integrations, and
full benchmark orchestration are not bundled. Software tests do not reproduce
manuscript accuracy tables. Dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
