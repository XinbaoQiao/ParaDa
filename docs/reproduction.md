# Reproduce ParaDa from images and pretrained models

The input commands connect public models and local datasets to the existing
ParaDa tensor CLI. Baselines are outside this release. All paths below are
relative to the checkout; images, weights and generated features stay outside Git.

This is a **reproduction variant**, not a verified replay of published accuracy
tables. The input readers and sampling rules are preserved, while source text is
encoded from the complete public vocabulary and fresh MLP initialization is now
seeded. Historical runs reused older source tensors and checkpoints. Identical
model names or seeds alone do not establish historical numerical equivalence.

## Install and obtain inputs

```bash
python -m pip install -e ".[dev,inputs]"
python -m parada.prepare catalog
python -m parada.prepare fetch-vocabulary --output data/vocabulary
```

`fetch-vocabulary` downloads only four small, revision-pinned class-name and
WordNet-ID files. It checks their SHA-256 before writing. No source images are
needed to train the source MLP. Model loading never downloads implicitly.

| Profile | Public repository | Revision | Classifier shape |
| --- | --- | --- | --- |
| `vit_tiny` | [timm/vit_tiny_patch16_224.augreg_in21k](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k) | `3d5f75e2fe58abe541d5651356278a1df3fd3ab3` | 21843 × 192 |
| `convnext` | [timm/convnext_base.fb_in22k](https://huggingface.co/timm/convnext_base.fb_in22k) | `9afd4146adabf1b6e861d0f1ac2e7baa7c8dd1b7` | 21841 × 1024 |
| `beit` | [timm/beit_base_patch16_224.in22k_ft_in22k](https://huggingface.co/timm/beit_base_patch16_224.in22k_ft_in22k) | `9b9827a7c99ab763a12630f2f7d913763cff4152` | 21841 × 768 |

Each file is `model.safetensors`. Full checksums, preprocessing and vocabulary
bindings are in [model_catalog.json](../src/parada/model_catalog.json).
For example, obtain the ViT-Tiny model using the Hugging Face CLI:

```bash
hf download timm/vit_tiny_patch16_224.augreg_in21k model.safetensors --revision 3d5f75e2fe58abe541d5651356278a1df3fd3ab3 --local-dir data/models/vit_tiny
```

Obtain the official [CLIP ViT-B/32 weights](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)
as `data/models/ViT-B-32.pt`. The loader checks SHA-256
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
before loading. The `inputs` extra pins the official CLIP implementation revision.

All three vision models use frozen timm evaluation preprocessing with bicubic
interpolation. ViT-Tiny and BEiT use crop ratio 0.9 and mean/std 0.5. ConvNeXt
uses crop ratio 0.875 and ImageNet mean/std. Features are the input to the final
linear classifier, not arbitrary intermediate tokens or logits. Model weights,
feature dimensions and preprocessing are checked together.

## Complete CIFAR-10 example

Download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) separately. One option:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data/cifar10', train=True, download=True)"

python -m parada.prepare source --profile vit_tiny --checkpoint data/models/vit_tiny/model.safetensors --clip-checkpoint data/models/ViT-B-32.pt --vocabulary data/vocabulary --output outputs/source-vit.safetensors --device cuda:0

python -m parada.prepare index --dataset cifar10 --data-root data --output outputs/cifar10-index.json
python -m parada.prepare dataset --dataset cifar10 --data-root data --profile vit_tiny --checkpoint data/models/vit_tiny/model.safetensors --output outputs/cifar10-vit --device cuda:0

python -m parada.prepare target --cache outputs/cifar10-vit --descriptions outputs/cifar10-descriptions.json --clip-checkpoint data/models/ViT-B-32.pt --output outputs/cifar10-text.safetensors --device cuda:0

python -m parada train --source outputs/source-vit.safetensors --checkpoint outputs/source-mlp-vit-seed42 --seed 42 --device cuda:0

python -m parada.prepare episode --cache outputs/cifar10-vit --source-inputs outputs/source-vit.safetensors --target outputs/cifar10-text.safetensors --k 1 --seed 42 --regime fullway --output outputs/cifar10-k1-seed42

python -m parada adapt --source outputs/source-vit.safetensors --checkpoint outputs/source-mlp-vit-seed42 --target outputs/cifar10-k1-seed42/target.safetensors --clients outputs/cifar10-k1-seed42 --k 1 --seed 42 --output outputs/cifar10-k1-seed42/classifier.safetensors --device cuda:0
python -m parada predict --features outputs/cifar10-k1-seed42/query.safetensors --classifier outputs/cifar10-k1-seed42/classifier.safetensors --output outputs/cifar10-k1-seed42/predictions.safetensors
python -m parada.prepare evaluate --predictions outputs/cifar10-k1-seed42/predictions.safetensors --labels outputs/cifar10-k1-seed42/labels.safetensors --output outputs/cifar10-k1-seed42/metrics.json
```

Before the `target` command, supply `outputs/cifar10-descriptions.json` using the
description workflow below. Descriptions are required external inputs.

For CPU execution, change `--device cuda:0` to `--device cpu` consistently. Full
feature extraction and 500-epoch source training can be expensive. Do not compare
CPU and CUDA outputs as if they were bitwise identical. No full-dataset training
or extraction was performed as part of the release's lightweight software tests.

To bind a previously reviewed dataset index, pass its `index_sha256` value as
`dataset --expected-index VALUE`. Feature caches record exact sample order,
labels, checkpoint, preprocessing, environment and tensor-file hashes. Reuse the
same cache, descriptions and source checkpoint across target configurations.
Use new output paths: prepared inputs and episode outputs are write-once.

## Dataset layouts and split contracts

`--data-root data` means the following directories. The readers do not download,
replace missing split files, or move a dataset between train and test implicitly.

| Dataset ID / directory | Required data and split |
| --- | --- |
| `cifar10` / `cifar10` | Official `cifar-10-batches-py`; native train/test order |
| `cifar100` / `cifar100` | Official `cifar-100-python`; native train/test order |
| `cub2011` / `cub` | `CUB_200_2011/{images,images.txt,image_class_labels.txt,train_test_split.txt,classes.txt}`; official split |
| `flowers102` / `flowers102` | `flowers-102/{jpg,setid.mat,imagelabels.mat}`; `trnid` support pool and `tstid` queries, validation excluded |
| `food101` / `food101` | `food-101/{images,meta/train.json,meta/test.json}`; preserve metadata order |
| `stanforddogs` / `StanfordDogs` | `Images`, `train_list.mat`, `test_list.mat`; annotation-list order |
| `dtd` / `dtd` | torchvision DTD layout; partition 1 train+val versus test |
| `oxford_pets` / `oxford_pets` | torchvision Oxford-IIIT Pet layout; trainval versus test |
| `artbench` / `artbench` | Extracted ImageFolder train/test; fixed class-folder order |
| `waterbirds` / `waterbirds` | `waterbirds_v1.0/metadata.csv` and images; split 0 support, split 2 query |
| `spawrious` / `spawrious` | Pinned o2o-hard directory layout; historical selection depends on directory enumeration order |
| `eurosat` / `eurosat` | Official RGB class directories; stratified 90/10, seed 42 |
| `resisc45` / `resisc45` | Images plus frozen `resisc45-train.txt`, `resisc45-val.txt`, `resisc45-test.txt` lists; missing/mismatched lists are an error |
| `places365` / `places365` | Official validation metadata/images, capped at first 50 images per class; then derived 80/20 split, seed 3407 |
| `ham10000` / `ham10000` | Exported HF training-pool metadata and images; derived stratified 90/10 split, seed 3407 |
| `camelyon17` / `camelyon17` | WILDS metadata/images, OOD-validation center 1 only; derived 80/20 split, seed 3407 |
| `oct2017` / `oct2017` | OCT2017 train images plus `randomized_patient_id_group_assignment.jsonl`; frozen validation pool, then derived 80/20 split, seed 3407 |

The last four derived splits are sample-level project protocols. They are not
claims of standard benchmark splits or patient-disjoint support/query sampling.
Spawrious historical replay requires its original ordered input index; matching
files on another filesystem is insufficient. A new index identifies a new input
variant. Dataset access and redistribution remain subject to upstream terms.

For RESISC45, fetch the exact revision-pinned lists (18,900 train, 6,300 validation,
6,300 test) into the image root; each downloaded file is checked against its
historical SHA-256:

```bash
python -m parada.prepare fetch-splits --dataset resisc45 --output data/resisc45
```

The historical OCT2017 profile is **blocked for a fresh public-input reproduction**.
It additionally requires `metadata/randomized_patient_id_group_assignment.jsonl`
beside `OCT2017/train`. This project-generated assignment is not part of the
official image archive and is not bundled in this code-only release. Its SHA-256 is
`da410b18537a45b572463a88add891d7c42afc4e7bc5aa22e4392b2f58e77d9f`.
The reader requires these exact bytes, selects the frozen validation groups, then
constructs the stated sample-level 80/20 split. It does not use the archive's
original test set. A matching generator is unavailable in this package; do not
substitute a random split and label it as the historical protocol.

## Descriptions, episodes and metrics

Generated descriptions are not distributed. The code emits structured requests
using the original prompt/schema and batches of at most 50 ordered class names:

```bash
python -m parada.prepare description-prompts --index outputs/cifar10-index.json --output outputs/cifar10-requests
```

Run these request JSON files with your own text-generation service and save each
response locally. Requests identify the historical model `gpt-4o-2024-11-20`;
if it is unavailable, explicitly record any model change as a new input variant.
The package makes no generation API calls and requires no API credentials.
Merge responses in batch order and validate every class name and description:

```bash
python -m parada.prepare collect-descriptions --index outputs/cifar10-index.json --responses outputs/cifar10-response.json --output outputs/cifar10-descriptions.json
```

For datasets over 50 classes, list all response files in order after `--responses`.
Responses may be Chat Completions envelopes or decoded objects containing a
`descriptions` array. Only class names enter generation; images, labels and metrics
are not sent. Every new corpus has its own file hash. Generation is not
deterministic, so it cannot recreate historical text bytes or guarantee historical
accuracy. See [input schema](inputs.md) for the required three fields.

Source class names are encoded as bare names, normalized in CLIP's native dtype,
then converted to FP32. Target descriptions use the template in the input schema;
their embeddings stay raw until the MLP. Each target prompt must fit the tokenizer
without truncation. The target class-name list must exactly match the dataset
cache; there is no positional fallback for an unrecognized label order.

Global K support selection occurs **before** distribution to ten fixed owners.
The full training pool is partitioned using Dirichlet alpha 0.1, seed 3407.
For class c and episode e, support RNG seed is `seed + 1009*c + 10007*e`, so
support is nested across K. Empty clients remain empty.

`--regime fullway` uses all target classes and held-out queries. `--regime 5way
--episode e` selects the corresponding one of 200 episodes, five sorted classes,
and one held-out query per class. Classes and queries are shared across K. Repeat
e=0,...,199 with distinct directories; pass `--episode-seed seed+10007*e` to
`parada adapt`. Each episode starts from the same source checkpoint and zero
residual. Use seeds 42--46 separately and report seed means with sample standard
deviation only after all intended episodes have completed on the same backend.

For K=0, `episode` creates no client files; omit `adapt --clients`. K=1 and K=5
retain the five-round budgets in [implementation notes](../ALIGNMENT.md).
`evaluate` reports accuracy and balanced accuracy as fractions; choose the
dataset's declared metric before inspecting outcomes. Query labels are never
passed to source training or adaptation.

## Current evidence boundary

After installing the input dependencies, a bounded real-input check is available:

```bash
python tools/verify_inputs.py --profile vit_tiny --checkpoint data/models/vit_tiny/model.safetensors --clip-checkpoint data/models/ViT-B-32.pt --data-root data --dataset cifar10 --descriptions outputs/cifar10-descriptions.json --output outputs/input-check.json --device cpu
```

It authenticates both weight files, indexes the complete dataset, checks three
images from each split against the model's original classifier logits, and encodes
the fixed class descriptions. It does not train the source MLP or report accuracy.

The local test suite exercises real-image dataset fixtures, split determinism,
model checksum rejection, raw/native text normalization, sampling, checkpoint
seed reproducibility, and the complete small-fixture preparation-to-score path.
Fixture encoders and shortened source training establish software behavior only.
A separate CPU diagnostic authenticated the pinned ViT-Tiny, ConvNeXt and CLIP
weights, indexed all 60,000 CIFAR-10 examples, and decoded six images per vision
profile. Reconstructed classifier logits matched the original model exactly on
those samples. This check used Torch 2.13 / torchvision 0.28 / timm 1.0.28,
whereas the declared installation profile and local fixture tests use Torch 2.10 /
torchvision 0.25. It is input-compatibility evidence, not a full installation or
benchmark test of the declared profile. Real-weight BEiT validation is still pending.

Strict historical replay additionally needs the exact source checkpoint/tensors,
all frozen descriptions, the original split/order metadata and a full evaluation
on the declared backend. [Machine-readable status](reproduction-status.json)
records these remaining gaps. Baseline completion is independent of this package.
