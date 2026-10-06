from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from torchvision.datasets.imagenet import load_meta_file

from parada._inputs.data.bmpl_input_registry import REFERENCE_DATA_SUBDIRS, _build_manifest
from parada._inputs.data.bmpl_reference_manifests import (
    CanonicalTrainManifest,
    CanonicalTrainSample,
)
from parada._inputs.data.bmpl_reference_records import OrderedRecord, read_bmpl_ordered_records
from parada._inputs.data.bmpl_tier_a_runtime import (
    TIER_A_DATASETS,
    TierASample,
    load_tier_a_dataset,
    resolve_tier_a_class_names,
)
from parada._inputs.experiments.bmpl_fed100_matrix import DATASET_IDS
from parada._inputs.hashing import canonical_sha256, file_sha256


@dataclass(frozen=True, slots=True)
class AllDatasetManifest:
    dataset_id: str
    split: str
    class_names: tuple[str, ...]
    class_order_hash: str
    records: tuple[TierASample, ...]
    asset_identity: tuple[tuple[str, str], ...]
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class AllDatasetBundle:
    dataset_id: str
    class_names: tuple[str, ...]
    train_dataset: Any
    test_dataset: Any
    train_manifest: AllDatasetManifest
    test_manifest: AllDatasetManifest
    b0_train_manifest_hash: str
    canonical_train_manifest: CanonicalTrainManifest


@dataclass(slots=True)
class PathImageDataset(Dataset[tuple[str, Any, int]]):
    records: tuple[TierASample, ...]
    image_root: Path
    transform: Callable[[Image.Image], Any] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, Any, int]:
        row = self.records[index]
        path = (self.image_root / row.relative_path_or_native_index).resolve()
        if not path.is_relative_to(self.image_root.resolve()) or not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        value = self.transform(image) if self.transform is not None else image
        return (row.sample_id, value, row.label)


def load_all_dataset(
    dataset_id: str,
    reference_data_root: str | Path,
    *,
    transform: Callable[[Image.Image], Any] | None,
    imagenet100_class_indices: Sequence[int],
) -> AllDatasetBundle:
    """Build complete train/test datasets without importing the reference repository."""
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"unsupported frozen dataset: {dataset_id}")
    root = Path(reference_data_root).resolve()
    dataset_root = root / REFERENCE_DATA_SUBDIRS[dataset_id]
    if dataset_id in TIER_A_DATASETS:
        full_train = read_bmpl_ordered_records(dataset_id, dataset_root)
        bundle = load_tier_a_dataset(
            dataset_id, root, support_positions=range(len(full_train.records)), transform=transform
        )
        class_names = bundle.class_names
        train_manifest = _manifest(
            dataset_id,
            "train",
            class_names,
            bundle.support_records,
            full_train.dataset_asset_identity,
        )
        test_manifest = _manifest(dataset_id, "test", class_names, bundle.test_records, ())
        return AllDatasetBundle(
            dataset_id=dataset_id,
            class_names=class_names,
            train_dataset=bundle.support_dataset,
            test_dataset=bundle.test_dataset,
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            b0_train_manifest_hash=_build_manifest(full_train).manifest_hash,
            canonical_train_manifest=_build_manifest(full_train),
        )
    class_names, imagenet100_wnids = resolve_all_class_names(
        dataset_id, dataset_root, imagenet100_class_indices=imagenet100_class_indices
    )
    train_ordered = read_bmpl_ordered_records(
        dataset_id, dataset_root, imagenet100_wnids=imagenet100_wnids or None
    )
    canonical_train = _build_manifest(train_ordered)
    train_records = tuple(_from_canonical(row) for row in canonical_train.rows)
    test_ordered, test_identity = _read_test_records(
        dataset_id, dataset_root, imagenet100_wnids=imagenet100_wnids
    )
    test_records = tuple(
        (_sample(dataset_id, "test", position, row) for position, row in enumerate(test_ordered))
    )
    train_manifest = _manifest(
        dataset_id, "train", class_names, train_records, train_ordered.dataset_asset_identity
    )
    test_manifest = _manifest(dataset_id, "test", class_names, test_records, test_identity)
    train_root = _image_root(dataset_id, dataset_root, "train")
    test_root = _image_root(dataset_id, dataset_root, "test")
    return AllDatasetBundle(
        dataset_id=dataset_id,
        class_names=class_names,
        train_dataset=PathImageDataset(train_records, train_root, transform),
        test_dataset=PathImageDataset(test_records, test_root, transform),
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        b0_train_manifest_hash=canonical_train.manifest_hash,
        canonical_train_manifest=canonical_train,
    )


def resolve_dataset_class_names(
    dataset_id: str, reference_data_root: str | Path, *, imagenet100_class_indices: Sequence[int]
) -> tuple[str, ...]:
    root = Path(reference_data_root).resolve()
    dataset_root = root / REFERENCE_DATA_SUBDIRS[dataset_id]
    if dataset_id in TIER_A_DATASETS:
        return resolve_tier_a_class_names(dataset_id, dataset_root)
    names, _ = resolve_all_class_names(
        dataset_id, dataset_root, imagenet100_class_indices=imagenet100_class_indices
    )
    return names


def resolve_all_class_names(
    dataset_id: str, dataset_root: str | Path, *, imagenet100_class_indices: Sequence[int]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = Path(dataset_root)
    if dataset_id in {"imagenet-1k", "imagenet-100"}:
        wnid_to_classes, _ = load_meta_file(root)
        wnids = tuple(sorted(path.name for path in (root / "train").iterdir() if path.is_dir()))
        if len(wnids) != 1000:
            raise ValueError("ImageNet train root must contain exactly 1000 class folders")
        if dataset_id == "imagenet-1k":
            return (tuple(str(wnid_to_classes[wnid][0]) for wnid in wnids), ())
        indices = tuple(int(value) for value in imagenet100_class_indices)
        if len(indices) != 100 or len(set(indices)) != 100:
            raise ValueError("ImageNet-100 requires 100 unique frozen class indices")
        selected = tuple(wnids[index] for index in indices)
        return (tuple(str(wnid_to_classes[wnid][0]) for wnid in selected), selected)
    if dataset_id == "waterbirds":
        return (("Land Bird", "Water Bird"), ())
    if dataset_id == "cub2011":
        classes_path = root / "CUB_200_2011" / "classes.txt"
        names = []
        for line in classes_path.read_text(encoding="utf-8").splitlines():
            _, raw = line.split(maxsplit=1)
            value = raw.split(".", 1)[-1].replace("_", " ")
            names.append(value)
        return (tuple(names), ())
    if dataset_id == "flowers102":
        dataset = Flowers102(root=root, split="test", download=False)
        return (tuple(str(name).replace("?", "") for name in dataset.classes), ())
    raise ValueError(f"class names for {dataset_id} are supplied by the Tier-A runtime")


def _read_test_records(
    dataset_id: str, root: Path, *, imagenet100_wnids: Sequence[str]
) -> tuple[tuple[OrderedRecord, ...], tuple[tuple[str, str], ...]]:
    if dataset_id in {"imagenet-1k", "imagenet-100"}:
        val_root = root / "val"
        all_wnids = tuple(sorted(path.name for path in (root / "train").iterdir() if path.is_dir()))
        selected = tuple(imagenet100_wnids) if dataset_id == "imagenet-100" else all_wnids
        label_by_wnid = {wnid: index for index, wnid in enumerate(selected)}
        imagenet_rows = _imagefolder_records(val_root, label_by_wnid)
        identity = (
            (
                "val_inventory_sha256",
                canonical_sha256(
                    [(r.relative_path_or_native_index, r.label) for r in imagenet_rows]
                ),
            ),
        )
        return (imagenet_rows, identity)
    if dataset_id == "waterbirds":
        base = root / "waterbirds_v1.0"
        metadata = base / "metadata.csv"
        waterbird_rows: list[OrderedRecord] = []
        with metadata.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("split")) == "2":
                    waterbird_rows.append(OrderedRecord(str(row["img_filename"]), int(row["y"])))
        return (tuple(waterbird_rows), (("metadata_sha256", file_sha256(metadata)),))
    if dataset_id == "cub2011":
        base = root / "CUB_200_2011"
        paths = _space_table(base / "images.txt")
        labels = _space_table(base / "image_class_labels.txt")
        splits = _space_table(base / "train_test_split.txt")
        cub_rows = tuple(
            OrderedRecord(paths[key], int(labels[key]) - 1) for key in paths if splits[key] == "0"
        )
        return (
            cub_rows,
            tuple(
                (path.name, file_sha256(path))
                for path in (
                    base / "images.txt",
                    base / "image_class_labels.txt",
                    base / "train_test_split.txt",
                )
            ),
        )
    if dataset_id == "flowers102":
        base = root / "flowers-102"
        test_ids = loadmat(base / "setid.mat", squeeze_me=True)["tstid"].tolist()
        labels = loadmat(base / "imagelabels.mat", squeeze_me=True)["labels"].tolist()
        flower_rows = tuple(
            OrderedRecord(f"jpg/image_{int(image_id):05d}.jpg", int(labels[int(image_id) - 1]) - 1)
            for image_id in test_ids
        )
        return (
            flower_rows,
            (
                ("setid.mat", file_sha256(base / "setid.mat")),
                ("imagelabels.mat", file_sha256(base / "imagelabels.mat")),
            ),
        )
    raise ValueError(f"unsupported non-Tier-A test dataset: {dataset_id}")


def _image_root(dataset_id: str, root: Path, split: str) -> Path:
    if dataset_id in {"imagenet-1k", "imagenet-100"}:
        return root / ("train" if split == "train" else "val")
    if dataset_id == "waterbirds":
        return root / "waterbirds_v1.0"
    if dataset_id == "cub2011":
        return root / "CUB_200_2011" / "images"
    if dataset_id == "flowers102":
        return root / "flowers-102"
    raise ValueError(dataset_id)


def _imagefolder_records(root: Path, class_to_label: dict[str, int]) -> tuple[OrderedRecord, ...]:
    rows: list[OrderedRecord] = []
    for class_name in sorted(class_to_label):
        for path in sorted(item for item in (root / class_name).rglob("*") if item.is_file()):
            rows.append(
                OrderedRecord(path.relative_to(root).as_posix(), class_to_label[class_name])
            )
    return tuple(rows)


def _space_table(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        table[key] = value
    return table


def _from_canonical(row: CanonicalTrainSample) -> TierASample:
    return TierASample(row.position, row.sample_id, row.label, row.relative_path_or_native_index)


def _sample(dataset_id: str, split: str, position: int, row: OrderedRecord) -> TierASample:
    relative = str(row.relative_path_or_native_index)
    suffix = canonical_sha256(
        {
            "dataset_id": dataset_id,
            "split": split,
            "position": position,
            "relative_path_or_native_index": relative,
        }
    )[:16]
    return TierASample(
        position, f"{dataset_id}/{split}/{position:08d}/{suffix}", int(row.label), relative
    )


def _manifest(
    dataset_id: str,
    split: str,
    class_names: Sequence[str],
    records: Sequence[TierASample],
    asset_identity: Sequence[tuple[str, str]],
) -> AllDatasetManifest:
    classes = tuple(class_names)
    rows = tuple(records)
    assets = tuple(((str(key), str(value)) for key, value in asset_identity))
    identity = {
        "dataset_id": dataset_id,
        "split": split,
        "class_names": classes,
        "records": [
            (row.position, row.sample_id, row.label, row.relative_path_or_native_index)
            for row in rows
        ],
        "asset_identity": assets,
    }
    return AllDatasetManifest(
        dataset_id,
        split,
        classes,
        canonical_sha256(classes),
        rows,
        assets,
        canonical_sha256(identity),
    )
