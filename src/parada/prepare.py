"""Portable model, dataset and episode preparation; no implicit downloads."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from safetensors.torch import load_file

from parada import source

CATALOG = json.loads(Path(__file__).with_name("model_catalog.json").read_text())
DATASETS = (
    "cifar10",
    "cifar100",
    "cub2011",
    "flowers102",
    "food101",
    "stanforddogs",
    "oxford_pets",
    "dtd",
    "artbench",
    "spawrious",
    "waterbirds",
    "resisc45",
    "eurosat",
    "places365",
    "ham10000",
    "camelyon17",
    "oct2017",
)
LEGACY = {
    "cifar10",
    "cifar100",
    "cub2011",
    "flowers102",
    "food101",
    "stanforddogs",
    "artbench",
    "spawrious",
    "waterbirds",
}


def verify_file(path, expected):
    path = Path(path)
    if not path.is_file() or source.file_sha256(path) != expected:
        raise ValueError(f"missing or mismatched input: {path.name}")
    return path


def environment():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "timm", "numpy", "safetensors", "Pillow", "scipy")
    }


def vision(profile, checkpoint, device):
    from timm.data.transforms_factory import create_transform

    from parada._inputs.models.vision import TimmVisionEncoder

    row = CATALOG["models"][profile]
    verify_file(checkpoint, row["sha256"])
    model = TimmVisionEncoder(
        row["model"],
        pretrained=False,
        device=device,
        checkpoint_path=checkpoint,
        expected_sha256=row["sha256"],
    )
    # Explicit transform avoids changes in library defaults.
    model.transform = create_transform(**row["preprocessing"], is_training=False)
    if tuple(model.classification_head_rows().shape) != (row["classes"], row["width"]):
        raise ValueError("checkpoint classifier shape differs from its profile")
    return model


def text_encoder(checkpoint, device):
    import clip

    verify_file(checkpoint, CATALOG["text"]["sha256"])
    model, _ = clip.load(str(checkpoint), device=device, jit=False)
    return clip, model.eval().requires_grad_(False)


@torch.inference_mode()
def encode_text(clip, model, texts, device, *, normalize=False, batch_size=64):
    rows = []
    for offset in range(0, len(texts), batch_size):
        # Source names follow the reference tokenizer; target prompts must fit.
        tokens = clip.tokenize(texts[offset : offset + batch_size], truncate=normalize).to(device)
        values = model.encode_text(tokens)
        if normalize:
            values = values / values.norm(dim=-1, keepdim=True)
        rows.append(values.float().cpu())
    return torch.cat(rows).contiguous()


def fetch_vocabulary(output):
    fetch_metadata(output, CATALOG["vocabularies"])


def fetch_metadata(output, entries):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, row in entries.items():
        target = output / name
        if target.exists():
            verify_file(target, row["sha256"])
            continue
        import hashlib

        with urlopen(row["url"], timeout=60) as response:
            payload = response.read(2 * 1024 * 1024)
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("metadata download hash mismatch")
        with target.open("xb") as handle:
            handle.write(payload)


def prepare_source(profile, checkpoint, clip_checkpoint, vocabulary, output, device):
    row = CATALOG["models"][profile]
    names = []
    for kind in ("classes", "wnids"):
        name = f"{row['vocabulary']}_{kind}.txt"
        path = verify_file(Path(vocabulary) / name, CATALOG["vocabularies"][name]["sha256"])
        values = path.read_text(encoding="utf-8").splitlines()
        if len(values) != row["classes"]:
            raise ValueError("source vocabulary does not align with classifier rows")
        if kind == "classes":
            names = values
    encoder = vision(profile, checkpoint, device)
    head = encoder.classification_head_rows()
    del encoder
    clip, text_model = text_encoder(clip_checkpoint, device)
    text = encode_text(clip, text_model, names, device, normalize=True)
    identity = {
        "profile": profile,
        "vision_sha256": row["sha256"],
        "text_sha256": CATALOG["text"]["sha256"],
        "vocabulary": row["vocabulary"],
        "preprocessing": row["preprocessing"],
        "source_encoding": "full-vocabulary-native-normalization",
        "device": str(device),
        "environment": environment(),
        "claim_tier": "reproduction-variant",
        "historical_tensor_parity": "unverified",
    }
    source.write_once_safetensors(
        Path(output),
        {"source_text": text, "source_visual": head},
        {"input_identity": json.dumps(identity)},
    )
    identity["tensor_file_sha256"] = source.file_sha256(Path(output))
    source.write_once_json(Path(str(output) + ".json"), identity)


def load_dataset(dataset, data_root, transform=None):
    from torch.utils.data import Subset

    from parada._inputs.data.bmpl_all_dataset_runtime import load_all_dataset
    from parada._inputs.data.formal_zero_shot_datasets import load_formal_dataset

    root = Path(data_root)
    if dataset not in DATASETS:
        raise ValueError("dataset is outside the released roster")
    if dataset in LEGACY:
        bundle = load_all_dataset(dataset, root, transform=transform, imagenet100_class_indices=())
        splits = {"train": bundle.train_dataset, "test": bundle.test_dataset}
        records = {"train": bundle.train_manifest.records, "test": bundle.test_manifest.records}
        policy = {
            "id": "canonical-metadata-order",
            "train_manifest": bundle.train_manifest.manifest_hash,
            "test_manifest": bundle.test_manifest.manifest_hash,
        }
        return bundle.class_names, splits, records, policy
    if dataset == "oct2017":
        row = CATALOG["local_split_files"][dataset]
        candidates = list((root / dataset).rglob(row["filename"]))
        if len(candidates) != 1:
            raise ValueError("require exactly one frozen OCT2017 assignment file")
        verify_file(candidates[0], row["sha256"])
    elif dataset == "resisc45":
        for name, row in CATALOG["split_files"][dataset].items():
            verify_file(root / dataset / name, row["sha256"])
    view = load_formal_dataset(
        dataset,
        project_root=root,
        external_data_roots={dataset: root / dataset},
        transform=transform,
    )
    if not len(view.splits.get("train") or ()):
        # Preserve the explicitly documented project-derived few-shot split.
        pool = view.splits[view.evaluation_split]
        pool_records = view.records[view.evaluation_split]
        labels = np.asarray([r.label for r in pool_records])
        rng = np.random.RandomState(3407)
        train, test = [], []
        fraction = 0.9 if dataset == "ham10000" else 0.8
        for label in sorted(set(labels.tolist())):
            rows = np.flatnonzero(labels == label).copy()
            if len(rows) < 2:
                raise ValueError("each derived-split class requires at least two examples")
            rng.shuffle(rows)
            cut = min(max(int(np.floor(fraction * len(rows))), 1), len(rows) - 1)
            train.extend(rows[:cut].tolist())
            test.extend(rows[cut:].tolist())
        indices = {"train": sorted(train), "test": sorted(test)}
        splits = {key: Subset(pool, values) for key, values in indices.items()}
        records = {key: tuple(pool_records[i] for i in values) for key, values in indices.items()}
        policy = {
            "id": "derived-stratified-pool",
            "fraction": fraction,
            "seed": 3407,
            "pool_policy": view.split_policy,
            "patient_disjoint": False,
        }
    else:
        splits = {key: view.splits[key] for key in ("train", "test")}
        records = {key: view.records[key] for key in splits}
        policy = view.split_policy
    return view.prompt_class_names, splits, records, {**policy, "raw_data": view.raw_data_binding}


def dataset_index(dataset, data_root, transform=None):
    names, splits, records, policy = load_dataset(dataset, data_root, transform)
    rows = {
        key: [{"id": row.sample_id, "label": int(row.label)} for row in records[key]]
        for key in ("train", "test")
    }
    for key, entries in rows.items():
        if len(entries) != len(splits[key]) or len({r["id"] for r in entries}) != len(entries):
            raise ValueError("dataset rows and sample identities differ or contain duplicates")
    if {r["id"] for r in rows["train"]} & {r["id"] for r in rows["test"]}:
        raise ValueError("training and query sample identities overlap")
    identity = {
        "dataset": dataset,
        "class_names": list(names),
        "splits": rows,
        "split_policy": policy,
    }
    identity["index_sha256"] = source.canonical_sha256(identity)
    return splits, identity


@torch.inference_mode()
def prepare_dataset(
    dataset, data_root, profile, checkpoint, output, device, batch_size, expected_index=None
):
    from torch.utils.data import DataLoader

    output = Path(output)
    if output.exists():
        raise FileExistsError("feature output already exists; choose a new directory")
    encoder = vision(profile, checkpoint, device)
    splits, identity = dataset_index(dataset, data_root, encoder.transform)
    if expected_index and identity["index_sha256"] != expected_index:
        raise ValueError("dataset split/order differs from the frozen index")
    identity.update(
        profile=profile,
        vision_sha256=CATALOG["models"][profile]["sha256"],
        preprocessing=CATALOG["models"][profile]["preprocessing"],
        environment=environment(),
    )
    files = {}
    for split, dataset_object in splits.items():
        features, labels, ids = [], [], []
        for batch_ids, images, y in DataLoader(
            dataset_object, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            features.append(encoder.encode(images))
            labels.append(torch.as_tensor(y, dtype=torch.int64))
            ids.extend(batch_ids)
        if ids != [row["id"] for row in identity["splits"][split]]:
            raise ValueError("feature extraction changed sample order")
        path = output / f"{split}.safetensors"
        source.write_once_safetensors(
            path,
            {"features": torch.cat(features), "labels": torch.cat(labels)},
            {"vision_sha256": identity["vision_sha256"]},
        )
        files[path.name] = source.file_sha256(path)
    identity["files"] = files
    identity["status"] = "complete"
    source.write_once_json(output / "dataset.json", identity)


def read_cache(path):
    path = Path(path)
    info = json.loads((path / "dataset.json").read_text())
    if info.get("status") != "complete" or set(info["files"]) != {
        "train.safetensors",
        "test.safetensors",
    }:
        raise ValueError("feature cache is incomplete")
    index = {key: info[key] for key in ("dataset", "class_names", "splits", "split_policy")}
    if source.canonical_sha256(index) != info["index_sha256"]:
        raise ValueError("feature cache index hash mismatch")
    values = {
        split: load_file(
            str(verify_file(path / f"{split}.safetensors", info["files"][f"{split}.safetensors"]))
        )
        for split in ("train", "test")
    }
    for split, data in values.items():
        if (
            set(data) != {"features", "labels"}
            or data["features"].ndim != 2
            or data["labels"].ndim != 1
            or data["features"].shape[0] != data["labels"].shape[0]
            or data["labels"].dtype != torch.int64
            or not torch.isfinite(data["features"]).all()
        ):
            raise ValueError("invalid feature cache tensors")
        if data["labels"].tolist() != [r["label"] for r in info["splits"][split]]:
            raise ValueError("cache labels differ from its index")
    return info, values


def prepare_target(cache, descriptions, clip_checkpoint, output, device):
    from parada.descriptions import validate_description_rows

    info, _ = read_cache(cache)
    rows = json.loads(Path(descriptions).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("descriptions")
    if not isinstance(rows, list) or [r.get("class_name") for r in rows] != info["class_names"]:
        raise ValueError("description class names/order differ from dataset")
    rows = validate_description_rows(rows, info["class_names"])
    texts = []
    for row in rows:
        views = [row[key] for key in ("appearance", "functionality", "environment")]
        if len(set(views)) != 3 or any(
            not isinstance(v, str) or not 8 <= len(v.split()) <= 18 for v in views
        ):
            raise ValueError("require three distinct descriptions, each 8-18 words")
        texts.extend(f"A photo of a {row['class_name']}: {v}" for v in views)
    clip, model = text_encoder(clip_checkpoint, device)
    encoded = encode_text(clip, model, texts, device, normalize=False).reshape(len(rows), 3, 512)
    identity = {
        "class_names": info["class_names"],
        "text_sha256": CATALOG["text"]["sha256"],
        "description_sha256": source.file_sha256(Path(descriptions)),
        "normalization": "raw",
        "environment": environment(),
        "device": str(device),
    }
    source.write_once_safetensors(
        Path(output), {"target_views": encoded}, {"input_identity": json.dumps(identity)}
    )
    identity["tensor_file_sha256"] = source.file_sha256(Path(output))
    source.write_once_json(Path(str(output) + ".json"), identity)


def select_episode(train_y, test_y, classes, k, seed, regime, episode):
    if not 0 <= k <= 10 or not 0 <= episode < 200:
        raise ValueError("K must be 0..10 and episode index 0..199")
    if regime == "fullway":
        if episode != 0:
            raise ValueError("fullway has only episode zero")
        class_ids = tuple(range(classes))
        query = torch.arange(len(test_y))
    elif regime == "5way" and classes >= 5:
        rng = np.random.default_rng(seed + episode * 10007)
        class_ids = tuple(sorted(int(c) for c in rng.choice(classes, 5, replace=False)))
        query = []
        for c in class_ids:
            pool = np.flatnonzero(test_y.numpy() == c)
            if not len(pool):
                raise ValueError("episode class has no held-out query")
            query.append(int(np.random.default_rng(seed + episode * 10007 + c * 1009).choice(pool)))
        query = torch.tensor(query, dtype=torch.int64)
    else:
        raise ValueError("5way requires at least five classes")
    support = []
    for c in class_ids:
        candidates = np.flatnonzero(train_y.numpy() == c)
        if len(candidates) < k:
            raise ValueError("class lacks global K support")
        rng = np.random.default_rng(seed + c * 1009 + episode * 10007)
        support.extend(rng.permutation(candidates)[:k].tolist())
    return class_ids, torch.tensor(support, dtype=torch.int64), query


def prepare_episode(cache, source_inputs, target, output, k, seed, regime, episode):
    from parada._inputs.data.partitions import split_dirichlet

    info, arrays = read_cache(cache)
    si = json.loads(Path(str(source_inputs) + ".json").read_text())
    ti = json.loads(Path(str(target) + ".json").read_text())
    verify_file(source_inputs, si["tensor_file_sha256"])
    verify_file(target, ti["tensor_file_sha256"])
    if (
        si["vision_sha256"] != info["vision_sha256"]
        or si["preprocessing"] != info["preprocessing"]
        or ti["text_sha256"] != si["text_sha256"]
        or ti["class_names"] != info["class_names"]
    ):
        raise ValueError("source, target and image cache have different feature bases/class order")
    out = Path(output)
    if out.exists():
        raise FileExistsError("episode output already exists")
    train, test = arrays["train"], arrays["test"]
    class_ids, support, query = select_episode(
        train["labels"], test["labels"], len(info["class_names"]), k, seed, regime, episode
    )
    ids = [r["id"] for r in info["splits"]["train"]]
    partitions = split_dirichlet(
        ids, train["labels"].tolist(), num_clients=10, alpha=0.1, seed=3407, minimum_size=1
    )
    owner = {sample: i for i, name in enumerate(sorted(partitions)) for sample in partitions[name]}
    map_label = {c: i for i, c in enumerate(class_ids)}
    support_labels = torch.tensor(
        [map_label[int(train["labels"][i])] for i in support], dtype=torch.int64
    )
    if k:
        for i in range(10):
            positions = [j for j, index in enumerate(support.tolist()) if owner[ids[index]] == i]
            index = torch.tensor(positions, dtype=torch.int64)
            source.write_once_safetensors(
                out / f"client-{i}.safetensors",
                {
                    "support_features": train["features"][support[index]],
                    "support_labels": support_labels[index],
                },
                {},
            )
    views = load_file(str(target))["target_views"][list(class_ids)]
    source.write_once_safetensors(out / "target.safetensors", {"target_views": views}, {})
    source.write_once_safetensors(
        out / "query.safetensors", {"features": test["features"][query]}, {}
    )
    source.write_once_safetensors(
        out / "labels.safetensors",
        {
            "labels": torch.tensor(
                [map_label[int(test["labels"][i])] for i in query], dtype=torch.int64
            )
        },
        {},
    )
    source.write_once_json(
        out / "episode.json",
        {
            "dataset": info["dataset"],
            "regime": regime,
            "k": k,
            "seed": seed,
            "episode": episode,
            "episode_seed": seed + 10007 * episode,
            "class_ids": list(class_ids),
            "index_sha256": info["index_sha256"],
            "support_ids": [ids[i] for i in support.tolist()],
            "query_ids": [info["splits"]["test"][i]["id"] for i in query.tolist()],
            "partition_sha256": source.canonical_sha256(partitions),
            "source_sha256": si["tensor_file_sha256"],
            "target_sha256": ti["tensor_file_sha256"],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    vocabulary = sub.add_parser("fetch-vocabulary")
    vocabulary.add_argument("--output", type=Path, required=True)
    split_files = sub.add_parser("fetch-splits")
    split_files.add_argument("--dataset", choices=CATALOG["split_files"], required=True)
    split_files.add_argument("--output", type=Path, required=True)
    prompts = sub.add_parser("description-prompts")
    prompts.add_argument("--index", type=Path, required=True)
    prompts.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("collect-descriptions")
    merge.add_argument("--index", type=Path, required=True)
    merge.add_argument("--responses", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    for name in ("source", "dataset"):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=CATALOG["models"], required=True)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        if name == "source":
            p.add_argument("--clip-checkpoint", type=Path, required=True)
            p.add_argument("--vocabulary", type=Path, required=True)
        else:
            p.add_argument("--dataset", choices=DATASETS, required=True)
            p.add_argument("--data-root", type=Path, required=True)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--expected-index")
    index = sub.add_parser("index")
    index.add_argument("--dataset", choices=DATASETS, required=True)
    index.add_argument("--data-root", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    target = sub.add_parser("target")
    for arg in ("cache", "descriptions", "clip-checkpoint", "output"):
        target.add_argument("--" + arg, type=Path, required=True)
    target.add_argument("--device", default="cpu")
    ep = sub.add_parser("episode")
    for arg in ("cache", "source-inputs", "target", "output"):
        ep.add_argument("--" + arg, type=Path, required=True)
    ep.add_argument("--k", type=int, choices=range(11), required=True)
    ep.add_argument("--seed", type=int, default=42)
    ep.add_argument("--regime", choices=("fullway", "5way"), default="fullway")
    ep.add_argument("--episode", type=int, default=0)
    evaluate = sub.add_parser("evaluate")
    for arg in ("predictions", "labels", "output"):
        evaluate.add_argument("--" + arg, type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "catalog":
        print(json.dumps({**CATALOG, "datasets": DATASETS}, indent=2))
    elif command == "fetch-vocabulary":
        fetch_vocabulary(**args)
    elif command == "fetch-splits":
        fetch_metadata(args["output"], CATALOG["split_files"][args["dataset"]])
    elif command in {"description-prompts", "collect-descriptions"}:
        from parada.descriptions import (
            build_description_messages,
            description_response_schema,
            parse_description_response,
            validate_description_rows,
        )

        names = json.loads(args["index"].read_text())["class_names"]
        if command == "description-prompts":
            if args["output"].exists():
                raise FileExistsError("description request directory already exists")
            for offset in range(0, len(names), 50):
                source.write_once_json(
                    args["output"] / f"batch-{offset // 50:03d}.json",
                    {
                        "model": "gpt-4o-2024-11-20",
                        "messages": build_description_messages(names[offset : offset + 50]),
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "class_descriptions",
                                "strict": True,
                                "schema": description_response_schema(),
                            },
                        },
                    },
                )
        else:
            rows = []
            for path in args["responses"]:
                rows.extend(
                    parse_description_response(json.loads(path.read_text()))["descriptions"]
                )
            valid = validate_description_rows(rows, names)
            source.write_once_json(args["output"], {"descriptions": valid})
    elif command == "source":
        prepare_source(**args)
    elif command == "dataset":
        if args["batch_size"] <= 0:
            raise ValueError("batch size must be positive")
        prepare_dataset(**args)
    elif command == "index":
        output = args.pop("output")
        _, identity = dataset_index(**args)
        source.write_once_json(output, identity)
    elif command == "target":
        prepare_target(**args)
    elif command == "episode":
        prepare_episode(**args)
    else:
        pred = load_file(str(args["predictions"]))["predictions"]
        labels = load_file(str(args["labels"]))["labels"]
        if pred.shape != labels.shape or pred.ndim != 1 or not pred.numel():
            raise ValueError("predictions and labels require equal nonempty vectors")
        accuracy = float((pred == labels).float().mean())
        balanced = float(
            torch.stack([(pred[labels == c] == c).float().mean() for c in labels.unique()]).mean()
        )
        source.write_once_json(
            args["output"],
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "query_count": len(labels),
                "predictions_sha256": source.file_sha256(args["predictions"]),
                "labels_sha256": source.file_sha256(args["labels"]),
            },
        )


if __name__ == "__main__":
    main()
