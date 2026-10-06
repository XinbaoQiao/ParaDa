from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from parada._inputs.benchmarks.schema import is_portable_sample_id
from parada._inputs.hashing import canonical_sha256

DATASET_CLASS_COUNTS = {
    "artbench": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cub2011": 200,
    "flowers102": 102,
    "food101": 101,
    "imagenet-100": 100,
    "imagenet-1k": 1000,
    "spawrious": 4,
    "stanforddogs": 120,
    "waterbirds": 2,
}
SUPPORTED_DATASETS = frozenset(DATASET_CLASS_COUNTS)


@dataclass(frozen=True, slots=True)
class CanonicalTrainSample:
    position: int
    sample_id: str
    label: int
    relative_path_or_native_index: str


@dataclass(frozen=True, slots=True)
class CanonicalTrainManifest:
    dataset_id: str
    split: str
    builder_id: str
    rows: tuple[CanonicalTrainSample, ...]
    manifest_hash: str
    dataset_asset_identity: tuple[tuple[str, str], ...] = ()

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "builder_id": self.builder_id,
            "dataset_asset_identity": self.dataset_asset_identity,
            "rows": [asdict(row) for row in self.rows],
        }

    def computed_hash(self) -> str:
        return canonical_sha256(self.payload_without_hash())


def build_cifar10_train_manifest(
    labels: Sequence[int], *, dataset_asset_identity: Mapping[str, Any] | None = None
) -> CanonicalTrainManifest:
    return _build_cifar_train_manifest(
        dataset_id="cifar10",
        labels=labels,
        builder_id="bmpl-fedprototype-cifar10-train-native-index-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_cifar100_train_manifest(
    labels: Sequence[int], *, dataset_asset_identity: Mapping[str, Any] | None = None
) -> CanonicalTrainManifest:
    return _build_cifar_train_manifest(
        dataset_id="cifar100",
        labels=labels,
        builder_id="bmpl-fedprototype-cifar100-train-native-index-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_spawrious_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="spawrious",
        records=records,
        builder_id="bmpl-fedprototype-spawrious-train-spawrious-order-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_stanforddogs_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="stanforddogs",
        records=records,
        builder_id="bmpl-fedprototype-stanforddogs-train-mat-order-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_food101_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="food101",
        records=records,
        builder_id="bmpl-fedprototype-food101-train-torchvision-files-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_artbench_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="artbench",
        records=records,
        builder_id="bmpl-fedprototype-artbench-train-imagefolder-samples-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_imagenet_1k_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="imagenet-1k",
        records=records,
        builder_id="bmpl-fedprototype-imagenet-1k-train-imagefolder-filtered-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_imagenet_100_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="imagenet-100",
        records=records,
        builder_id="bmpl-fedprototype-imagenet-100-train-subset-order-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_waterbirds_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="waterbirds",
        records=records,
        builder_id="bmpl-fedprototype-waterbirds-train-wilds-order-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_cub2011_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="cub2011",
        records=records,
        builder_id="bmpl-fedprototype-cub2011-train-metadata-order-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def build_flowers102_train_manifest(
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    *,
    dataset_asset_identity: Mapping[str, Any] | None = None,
) -> CanonicalTrainManifest:
    return _build_path_record_train_manifest(
        dataset_id="flowers102",
        records=records,
        builder_id="bmpl-fedprototype-flowers102-train-torchvision-files-v1",
        dataset_asset_identity=dataset_asset_identity,
    )


def _build_cifar_train_manifest(
    *,
    dataset_id: str,
    labels: Sequence[int],
    builder_id: str,
    dataset_asset_identity: Mapping[str, Any] | None,
) -> CanonicalTrainManifest:
    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported BMPL reference train manifest dataset: {dataset_id}")
    rows = tuple(
        (
            CanonicalTrainSample(
                position=position,
                sample_id=_sample_id(dataset_id, position, f"native_index/{position:08d}"),
                label=_validate_label(label, DATASET_CLASS_COUNTS[dataset_id], position),
                relative_path_or_native_index=f"native_index/{position:08d}",
            )
            for position, label in enumerate(labels)
        )
    )
    _validate_train_rows(dataset_id=dataset_id, rows=rows, expected_count=len(labels))
    manifest = CanonicalTrainManifest(
        dataset_id=dataset_id,
        split="train",
        builder_id=builder_id,
        rows=rows,
        manifest_hash="",
        dataset_asset_identity=_normalize_asset_identity(dataset_asset_identity),
    )
    return CanonicalTrainManifest(
        dataset_id=manifest.dataset_id,
        split=manifest.split,
        builder_id=manifest.builder_id,
        rows=manifest.rows,
        manifest_hash=manifest.computed_hash(),
        dataset_asset_identity=manifest.dataset_asset_identity,
    )


def _build_path_record_train_manifest(
    *,
    dataset_id: str,
    records: Sequence[Mapping[str, Any] | Sequence[Any]],
    builder_id: str,
    dataset_asset_identity: Mapping[str, Any] | None,
) -> CanonicalTrainManifest:
    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported BMPL reference train manifest dataset: {dataset_id}")
    parsed = tuple(
        (_parse_ordered_record(record, position) for position, record in enumerate(records))
    )
    class_count = DATASET_CLASS_COUNTS[dataset_id]
    rows = tuple(
        (
            CanonicalTrainSample(
                position=position,
                sample_id=_sample_id(dataset_id, position, path_or_native_id),
                label=_validate_label(label, class_count, position),
                relative_path_or_native_index=path_or_native_id,
            )
            for position, (path_or_native_id, label) in enumerate(parsed)
        )
    )
    _validate_train_rows(dataset_id=dataset_id, rows=rows, expected_count=len(parsed))
    manifest = CanonicalTrainManifest(
        dataset_id=dataset_id,
        split="train",
        builder_id=builder_id,
        rows=rows,
        manifest_hash="",
        dataset_asset_identity=_normalize_asset_identity(dataset_asset_identity),
    )
    return CanonicalTrainManifest(
        dataset_id=manifest.dataset_id,
        split=manifest.split,
        builder_id=manifest.builder_id,
        rows=manifest.rows,
        manifest_hash=manifest.computed_hash(),
        dataset_asset_identity=manifest.dataset_asset_identity,
    )


def _validate_train_rows(
    *, dataset_id: str, rows: Sequence[CanonicalTrainSample], expected_count: int
) -> None:
    if not dataset_id:
        raise ValueError("dataset_id must be nonempty")
    if len(rows) != expected_count:
        raise ValueError(f"train manifest row count mismatch: expected {expected_count}")
    class_count = DATASET_CLASS_COUNTS.get(dataset_id)
    seen_sample_ids: set[str] = set()
    seen_source_keys: set[str] = set()
    for expected_position, row in enumerate(rows):
        if row.position != expected_position:
            raise ValueError(
                f"train manifest position drift at row {expected_position}: found {row.position}"
            )
        _validate_relative_or_native_index(row.relative_path_or_native_index)
        expected_sample_id = _sample_id(
            dataset_id, expected_position, row.relative_path_or_native_index
        )
        if row.sample_id != expected_sample_id:
            raise ValueError(
                f"train manifest sample_id drift at position "
                f"{expected_position}: expected {expected_sample_id}, found {row.sample_id}"
            )
        if not is_portable_sample_id(row.sample_id):
            raise ValueError(f"non-portable sample_id: {row.sample_id}")
        if row.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate train manifest sample_id: {row.sample_id}")
        seen_sample_ids.add(row.sample_id)
        if row.relative_path_or_native_index in seen_source_keys:
            raise ValueError(
                f"duplicate source relative_path_or_native_index "
                f"in train manifest: {row.relative_path_or_native_index}"
            )
        seen_source_keys.add(row.relative_path_or_native_index)
        if row.label < 0:
            raise ValueError(f"negative label at position {expected_position}")
        if class_count is not None and row.label >= class_count:
            raise ValueError(
                f"label at position {expected_position} is outside [0, {class_count}): {row.label}"
            )


def _validate_relative_or_native_index(value: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"non-portable relative_path_or_native_index: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"non-portable relative_path_or_native_index: {value}")


def _parse_ordered_record(
    record: Mapping[str, Any] | Sequence[Any], position: int
) -> tuple[str, int]:
    if isinstance(record, Mapping):
        raw_path = next(
            (
                record[key]
                for key in (
                    "relative_path_or_native_index",
                    "relative_path",
                    "path",
                    "filepath",
                    "native_id",
                )
                if key in record and record[key] is not None
            ),
            None,
        )
        if raw_path is None:
            raise ValueError(f"record at position {position} lacks a portable path/native key")
        raw_label = record.get("label", record.get("target"))
        if raw_label is None:
            raise ValueError(f"record at position {position} lacks label")
    elif isinstance(record, Sequence) and (not isinstance(record, str)) and (len(record) == 2):
        raw_path, raw_label = record
    else:
        raise ValueError(
            "ordered records must be mappings or two-item "
            "sequences of (relative_path_or_native_index, label)"
        )
    path_or_native_id = str(raw_path)
    _validate_relative_or_native_index(path_or_native_id)
    return (path_or_native_id, _require_int(raw_label, f"label at position {position}"))


def _sample_id(dataset_id: str, position: int, path_or_native_id: str) -> str:
    key_hash = canonical_sha256(
        {
            "dataset_id": dataset_id,
            "split": "train",
            "position": int(position),
            "relative_path_or_native_index": path_or_native_id,
        }
    )[:16]
    return f"{dataset_id}/train/{position:08d}/{key_hash}"


def _normalize_asset_identity(
    dataset_asset_identity: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if dataset_asset_identity is None:
        return ()
    if not isinstance(dataset_asset_identity, Mapping):
        raise ValueError("dataset_asset_identity must be a mapping")
    normalized: list[tuple[str, str]] = []
    for raw_key, raw_value in sorted(dataset_asset_identity.items()):
        key = str(raw_key)
        value = str(raw_value)
        if not key or not value:
            raise ValueError("dataset_asset_identity keys and values must be nonempty")
        if "\\" in key or "\n" in key or "\n" in value:
            raise ValueError("dataset_asset_identity must be portable single-line text")
        normalized.append((key, value))
    identity = tuple(normalized)
    _validate_asset_identity(identity)
    return identity


def _validate_asset_identity(identity: tuple[tuple[str, str], ...]) -> None:
    seen_keys: set[str] = set()
    for raw_key, raw_value in identity:
        key = str(raw_key)
        value = str(raw_value)
        if not key or not value:
            raise ValueError("dataset_asset_identity keys and values must be nonempty")
        if "\\" in key or "\n" in key or "\n" in value:
            raise ValueError("dataset_asset_identity must be portable single-line text")
        if key in seen_keys:
            raise ValueError(f"duplicate dataset_asset_identity key: {key}")
        seen_keys.add(key)


def _validate_label(label: int, class_count: int, position: int) -> int:
    value = _require_int(label, f"label at position {position}")
    if value < 0 or value >= class_count:
        raise ValueError(f"label at position {position} is outside [0, {class_count}): {value}")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value
