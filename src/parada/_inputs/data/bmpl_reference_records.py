from __future__ import annotations

import csv
import json
import os
import pickle
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from parada._inputs.data.bmpl_reference_manifests import DATASET_CLASS_COUNTS
from parada._inputs.hashing import canonical_sha256, file_sha256

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class MetadataReaderBlocked(RuntimeError):
    """Raised when a dataset needs a pre-exported ordered metadata file."""


@dataclass(frozen=True, slots=True)
class OrderedRecord:
    relative_path_or_native_index: str
    label: int


@dataclass(frozen=True, slots=True)
class OrderedRecordSet:
    dataset_id: str
    split: str
    reader_id: str
    records: tuple[OrderedRecord, ...]
    dataset_asset_identity: tuple[tuple[str, str], ...]

    def as_manifest_records(self) -> tuple[tuple[str, int], ...]:
        return tuple((row.relative_path_or_native_index, row.label) for row in self.records)


def read_bmpl_ordered_records(
    dataset_id: str,
    reference_data_root: str | Path,
    *,
    imagenet100_wnids: Sequence[str] | None = None,
    explicit_jsonl: str | Path | None = None,
) -> OrderedRecordSet:
    """Read frozen BMPL train records from a local reference dataset root."""
    root = Path(reference_data_root)
    if dataset_id == "cifar10":
        return read_cifar10_train_records(root)
    if dataset_id == "cifar100":
        return read_cifar100_train_records(root)
    if dataset_id == "artbench":
        return read_artbench_train_records(root)
    if dataset_id == "imagenet-1k":
        return read_imagenet_1k_train_records(root)
    if dataset_id == "imagenet-100":
        return read_imagenet_100_train_records(root, imagenet100_wnids=imagenet100_wnids)
    if dataset_id == "food101":
        return read_food101_train_records(root)
    if dataset_id == "flowers102":
        return read_flowers102_train_records(root, explicit_jsonl=explicit_jsonl)
    if dataset_id == "cub2011":
        return read_cub2011_train_records(root)
    if dataset_id == "waterbirds":
        return read_waterbirds_train_records(root)
    if dataset_id == "stanforddogs":
        return read_stanforddogs_train_records(root, explicit_jsonl=explicit_jsonl)
    if dataset_id == "spawrious":
        return read_spawrious_train_records(root, explicit_jsonl=explicit_jsonl)
    raise ValueError(f"unsupported BMPL ordered-record dataset: {dataset_id}")


def read_cifar10_train_records(root: str | Path) -> OrderedRecordSet:
    dataset_root = _dataset_root(Path(root), "cifar-10-batches-py")
    batch_names = tuple(f"data_batch_{index}" for index in range(1, 6))
    labels: list[int] = []
    source_files: list[Path] = []
    for batch_name in batch_names:
        batch_path = _require_file(dataset_root / batch_name)
        payload = _load_pickle(batch_path)
        labels.extend(_require_int_sequence(payload.get("labels"), f"{batch_name}.labels"))
        source_files.append(batch_path)
    return _native_index_record_set(
        dataset_id="cifar10",
        reader_id="bmpl-reference-records-cifar10-torchvision-pickle-v1",
        labels=labels,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-cifar10-torchvision-pickle-v1",
            source_files=source_files,
            extra={"label_order_sha256": canonical_sha256(labels)},
        ),
    )


def read_cifar100_train_records(root: str | Path) -> OrderedRecordSet:
    dataset_root = _dataset_root(Path(root), "cifar-100-python")
    train_path = _require_file(dataset_root / "train")
    payload = _load_pickle(train_path)
    labels = _require_int_sequence(payload.get("fine_labels"), "train.fine_labels")
    return _native_index_record_set(
        dataset_id="cifar100",
        reader_id="bmpl-reference-records-cifar100-torchvision-pickle-v1",
        labels=labels,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-cifar100-torchvision-pickle-v1",
            source_files=[train_path],
            extra={"label_order_sha256": canonical_sha256(labels)},
        ),
    )


def read_artbench_train_records(root: str | Path) -> OrderedRecordSet:
    train_root = _first_existing_dir(
        Path(root), ("artbench-10/train", "ArtBench-10/train", "train")
    )
    return _imagefolder_record_set(
        dataset_id="artbench",
        reader_id="bmpl-reference-records-artbench-imagefolder-v1",
        train_root=train_root,
    )


def read_imagenet_1k_train_records(root: str | Path) -> OrderedRecordSet:
    train_root = _first_existing_dir(Path(root), ("ILSVRC2012/train", "imagenet/train", "train"))
    return _imagefolder_record_set(
        dataset_id="imagenet-1k",
        reader_id="bmpl-reference-records-imagenet-1k-imagefolder-v1",
        train_root=train_root,
    )


def read_imagenet_100_train_records(
    root: str | Path, *, imagenet100_wnids: Sequence[str] | None
) -> OrderedRecordSet:
    if imagenet100_wnids is None:
        raise ValueError("imagenet-100 requires the frozen ImageNet-100 wnid class list")
    wnids = tuple(str(wnid) for wnid in imagenet100_wnids)
    if len(wnids) != 100 or len(set(wnids)) != 100:
        raise ValueError("imagenet-100 frozen wnid class list must contain 100 unique classes")
    train_root = _first_existing_dir(Path(root), ("ILSVRC2012/train", "imagenet/train", "train"))
    records = _imagefolder_records(
        train_root, class_to_label={wnid: index for index, wnid in enumerate(wnids)}
    )
    return _record_set(
        dataset_id="imagenet-100",
        reader_id="bmpl-reference-records-imagenet-100-imagefolder-filter-v1",
        records=records,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-imagenet-100-imagefolder-filter-v1",
            extra={
                "imagefolder_inventory_sha256": _directory_inventory_hash(train_root),
                "imagenet100_wnids_sha256": canonical_sha256(wnids),
            },
        ),
    )


def read_food101_train_records(root: str | Path) -> OrderedRecordSet:
    base = _first_existing_dir(Path(root), ("food-101", "."))
    meta_path = _require_file(base / "meta" / "train.json")
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise ValueError("Food101 train.json must contain a mapping")
    classes = tuple(sorted(str(name) for name in metadata))
    class_to_label = {name: index for index, name in enumerate(classes)}
    rows: list[OrderedRecord] = []
    for class_name, raw_paths in metadata.items():
        if not isinstance(raw_paths, list):
            raise ValueError(f"Food101 train.json class {class_name!r} must contain a list")
        label = class_to_label[str(class_name)]
        for raw_path in raw_paths:
            relative = f"images/{str(raw_path)}.jpg"
            _validate_portable_relative(relative)
            rows.append(OrderedRecord(relative, label))
    return _record_set(
        dataset_id="food101",
        reader_id="bmpl-reference-records-food101-torchvision-json-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-food101-torchvision-json-v1",
            source_files=[meta_path],
            extra={"class_order_sha256": canonical_sha256(classes)},
        ),
    )


def read_flowers102_train_records(
    root: str | Path, *, explicit_jsonl: str | Path | None = None
) -> OrderedRecordSet:
    if explicit_jsonl is not None:
        return read_explicit_jsonl_ordered_records(
            "flowers102",
            explicit_jsonl,
            reader_id="bmpl-reference-records-flowers102-explicit-jsonl-v1",
        )
    base = _first_existing_dir(Path(root), ("flowers-102", "."))
    setid = _require_file(base / "setid.mat")
    labels = _require_file(base / "imagelabels.mat")
    try:
        loadmat = import_module("scipy.io").loadmat
    except ModuleNotFoundError as error:
        raise MetadataReaderBlocked(
            "Flowers102 train order requires scipy.io.loadmat for "
            "setid.mat/imagelabels.mat or an explicit ordered JSONL export"
        ) from error
    split_ids = loadmat(setid, squeeze_me=True)["trnid"].tolist()
    raw_labels = loadmat(labels, squeeze_me=True)["labels"].tolist()
    image_id_to_label = {index: int(label) - 1 for index, label in enumerate(raw_labels, 1)}
    rows = [
        OrderedRecord(f"jpg/image_{int(image_id):05d}.jpg", image_id_to_label[int(image_id)])
        for image_id in split_ids
    ]
    return _record_set(
        dataset_id="flowers102",
        reader_id="bmpl-reference-records-flowers102-torchvision-mat-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-flowers102-torchvision-mat-v1",
            source_files=[setid, labels],
        ),
    )


def read_cub2011_train_records(root: str | Path) -> OrderedRecordSet:
    base = _first_existing_dir(Path(root), ("CUB_200_2011", "."))
    images_txt = _read_space_table(_require_file(base / "images.txt"), "images.txt")
    labels_txt = _read_space_table(
        _require_file(base / "image_class_labels.txt"), "image_class_labels.txt"
    )
    split_txt = _read_space_table(
        _require_file(base / "train_test_split.txt"), "train_test_split.txt"
    )
    rows: list[OrderedRecord] = []
    if set(labels_txt) != set(images_txt) or set(split_txt) != set(images_txt):
        raise ValueError("CUB2011 metadata tables must contain the same image IDs")
    for image_id in images_txt:
        if split_txt.get(image_id) != "1":
            continue
        relative = images_txt[image_id]
        label = int(labels_txt[image_id]) - 1
        rows.append(OrderedRecord(relative, label))
    return _record_set(
        dataset_id="cub2011",
        reader_id="bmpl-reference-records-cub2011-metadata-txt-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-cub2011-metadata-txt-v1",
            source_files=[
                base / "images.txt",
                base / "image_class_labels.txt",
                base / "train_test_split.txt",
            ],
        ),
    )


def read_waterbirds_train_records(root: str | Path) -> OrderedRecordSet:
    base = _first_existing_dir(Path(root), ("waterbirds_v1.0", "waterbirds", "."))
    metadata = _require_file(base / "metadata.csv")
    rows: list[OrderedRecord] = []
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = str(row.get("split", ""))
            if split not in {"0", "train"}:
                continue
            relative = row.get("img_filename") or row.get("filename") or row.get("path")
            if relative is None:
                raise ValueError("Waterbirds metadata.csv lacks img_filename/filename/path")
            rows.append(OrderedRecord(str(relative), _int_from_row(row, ("y", "label", "target"))))
    return _record_set(
        dataset_id="waterbirds",
        reader_id="bmpl-reference-records-waterbirds-metadata-csv-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-waterbirds-metadata-csv-v1", source_files=[metadata]
        ),
    )


def read_stanforddogs_train_records(
    root: str | Path, *, explicit_jsonl: str | Path | None = None
) -> OrderedRecordSet:
    if explicit_jsonl is not None:
        return read_explicit_jsonl_ordered_records(
            "stanforddogs",
            explicit_jsonl,
            reader_id="bmpl-reference-records-stanforddogs-explicit-jsonl-v1",
        )
    base = _first_existing_dir(Path(root), ("StanfordDogs", "."))
    split_file = _require_file(base / "train_list.mat")
    try:
        loadmat = import_module("scipy.io").loadmat
    except ModuleNotFoundError as error:
        raise MetadataReaderBlocked(
            "StanfordDogs train order requires scipy.io.loadmat "
            "for train_list.mat or an explicit ordered JSONL export"
        ) from error
    payload = loadmat(split_file, squeeze_me=True)
    if "annotation_list" not in payload or "labels" not in payload:
        raise ValueError("StanfordDogs train_list.mat lacks annotation_list/labels")
    names = _mat_vector(payload["annotation_list"], field="annotation_list")
    labels = _mat_vector(payload["labels"], field="labels")
    if len(names) != len(labels):
        raise ValueError("StanfordDogs annotation_list and labels lengths differ")
    rows = [
        OrderedRecord(f"{str(name)}.jpg", int(label) - 1)
        for name, label in zip(names, labels, strict=True)
    ]
    return _record_set(
        dataset_id="stanforddogs",
        reader_id="bmpl-reference-records-stanforddogs-train-list-mat-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-reference-records-stanforddogs-train-list-mat-v1",
            source_files=[split_file],
        ),
    )


def read_spawrious_train_records(
    root: str | Path, *, explicit_jsonl: str | Path | None = None
) -> OrderedRecordSet:
    if explicit_jsonl is not None:
        return read_explicit_jsonl_ordered_records(
            "spawrious",
            explicit_jsonl,
            reader_id="bmpl-reference-records-spawrious-explicit-jsonl-v1",
        )
    data_root = _first_existing_dir(Path(root), ("spawrious224", "."))
    class_names = ("bulldog", "dachshund", "labrador", "corgi")
    pinned_class_list = ("bulldog", "corgi", "dachshund", "labrador")
    class_to_label = {name: index for index, name in enumerate(pinned_class_list)}
    primary_locations = {
        "bulldog": "jungle",
        "dachshund": "mountain",
        "labrador": "snow",
        "corgi": "desert",
    }
    rows: list[OrderedRecord] = []
    ordered_directory_entries: dict[str, list[str]] = {}
    for phase, primary_limit, filler_limit in ((0, 3072, 96), (1, 2756, 412)):
        for class_name in class_names:
            rows.extend(
                _spawrious_directory_records(
                    data_root,
                    phase=phase,
                    location=primary_locations[class_name],
                    class_name=class_name,
                    label=class_to_label[class_name],
                    limit=primary_limit,
                    ordered_directory_entries=ordered_directory_entries,
                )
            )
        for class_name in class_names:
            rows.extend(
                _spawrious_directory_records(
                    data_root,
                    phase=phase,
                    location="beach",
                    class_name=class_name,
                    label=class_to_label[class_name],
                    limit=filler_limit,
                    ordered_directory_entries=ordered_directory_entries,
                )
            )
    reader_id = "bmpl-reference-records-spawrious-o2o-hard-os-listdir-v1"
    return _record_set(
        dataset_id="spawrious",
        reader_id=reader_id,
        records=rows,
        asset_identity=_identity(
            reader_id=reader_id,
            extra={
                "benchmark": "o2o_hard",
                "spawrious_package_commit": "651e90415bfb414a5b6579cb1f53b83e6bb8c165",
                "spawrious_class_list_sha256": canonical_sha256(pinned_class_list),
                "ordered_directory_entries_sha256": canonical_sha256(ordered_directory_entries),
            },
        ),
    )


def read_explicit_jsonl_ordered_records(
    dataset_id: str, path: str | Path, *, reader_id: str | None = None
) -> OrderedRecordSet:
    source = _require_file(Path(path))
    rows: list[OrderedRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            relative = _first_present(
                payload,
                ("relative_path_or_native_index", "relative_path", "path", "filepath", "native_id"),
                field=f"JSONL row {line_number} path/native key",
            )
            label = _int_from_row(payload, ("label", "target"))
            rows.append(OrderedRecord(str(relative), label))
    return _record_set(
        dataset_id=dataset_id,
        reader_id=reader_id or f"bmpl-reference-records-{dataset_id}-explicit-jsonl-v1",
        records=rows,
        asset_identity=_identity(
            reader_id=reader_id or f"bmpl-reference-records-{dataset_id}-explicit-jsonl-v1",
            source_files=[source],
        ),
    )


def _native_index_record_set(
    *,
    dataset_id: str,
    reader_id: str,
    labels: Sequence[int],
    asset_identity: tuple[tuple[str, str], ...],
) -> OrderedRecordSet:
    records = [
        OrderedRecord(f"native_index/{position:08d}", int(label))
        for position, label in enumerate(labels)
    ]
    return _record_set(
        dataset_id=dataset_id, reader_id=reader_id, records=records, asset_identity=asset_identity
    )


def _spawrious_directory_records(
    data_root: Path,
    *,
    phase: int,
    location: str,
    class_name: str,
    label: int,
    limit: int,
    ordered_directory_entries: dict[str, list[str]],
) -> list[OrderedRecord]:
    directory = data_root / str(phase) / location / class_name
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ordered_names = [
        name for name in os.listdir(directory) if name.endswith((".png", ".jpg", ".jpeg"))
    ]
    key = directory.relative_to(data_root).as_posix()
    ordered_directory_entries[key] = ordered_names
    return [
        OrderedRecord((Path(str(phase)) / location / class_name / name).as_posix(), label)
        for name in ordered_names[:limit]
    ]


def _mat_vector(value: Any, *, field: str) -> list[Any]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, list):
        raw = [raw]
    flattened: list[Any] = []
    for item in raw:
        current = item
        while isinstance(current, (list, tuple)) and len(current) == 1:
            current = current[0]
        item_method = getattr(current, "item", None)
        if callable(item_method):
            with suppress(ValueError):
                current = item_method()
        flattened.append(current)
    if not flattened:
        raise ValueError(f"MAT field {field} must be a nonempty vector")
    return flattened


def _imagefolder_record_set(
    *, dataset_id: str, reader_id: str, train_root: Path
) -> OrderedRecordSet:
    return _record_set(
        dataset_id=dataset_id,
        reader_id=reader_id,
        records=_imagefolder_records(train_root),
        asset_identity=_identity(
            reader_id=reader_id,
            extra={"imagefolder_inventory_sha256": _directory_inventory_hash(train_root)},
        ),
    )


def _imagefolder_records(
    train_root: Path, *, class_to_label: Mapping[str, int] | None = None
) -> tuple[OrderedRecord, ...]:
    class_names = tuple(
        class_name
        for class_name in sorted(path.name for path in train_root.iterdir() if path.is_dir())
        if class_to_label is None or class_name in class_to_label
    )
    if class_to_label is None:
        class_to_label = {class_name: index for index, class_name in enumerate(class_names)}
    if not class_names:
        raise FileNotFoundError(f"no class folders found under {train_root}")
    rows: list[OrderedRecord] = []
    for class_name in sorted(class_to_label):
        class_root = train_root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"required ImageFolder class is missing: {class_name}")
        for path in sorted(item for item in class_root.rglob("*") if item.is_file()):
            if path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            rows.append(
                OrderedRecord(
                    path.relative_to(train_root).as_posix(), int(class_to_label[class_name])
                )
            )
    return tuple(rows)


def _record_set(
    *,
    dataset_id: str,
    reader_id: str,
    records: Iterable[OrderedRecord],
    asset_identity: tuple[tuple[str, str], ...],
) -> OrderedRecordSet:
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError(f"unsupported BMPL ordered-record dataset: {dataset_id}")
    class_count = DATASET_CLASS_COUNTS[dataset_id]
    normalized: list[OrderedRecord] = []
    seen: set[str] = set()
    for position, row in enumerate(records):
        relative = str(row.relative_path_or_native_index)
        _validate_portable_relative(relative)
        if relative in seen:
            raise ValueError(f"duplicate ordered-record key: {relative}")
        seen.add(relative)
        label = _require_int(row.label, f"label at position {position}")
        if label < 0 or label >= class_count:
            raise ValueError(f"label at position {position} is outside [0, {class_count}): {label}")
        normalized.append(OrderedRecord(relative, label))
    if not normalized:
        raise ValueError(f"{dataset_id} ordered-record reader produced no train records")
    return OrderedRecordSet(
        dataset_id=dataset_id,
        split="train",
        reader_id=reader_id,
        records=tuple(normalized),
        dataset_asset_identity=tuple(sorted(asset_identity)),
    )


def _identity(
    *, reader_id: str, source_files: Iterable[Path] = (), extra: Mapping[str, str] | None = None
) -> tuple[tuple[str, str], ...]:
    values: dict[str, str] = {"reader_id": reader_id}
    for path in source_files:
        source = _require_file(path)
        safe_key = source.name.replace(".", "_").replace("-", "_")
        values[f"{safe_key}_sha256"] = file_sha256(source)
        values[f"{safe_key}_size_bytes"] = str(source.stat().st_size)
    if extra:
        values.update({str(key): str(value) for key, value in extra.items()})
    return tuple(sorted(values.items()))


def _directory_inventory_hash(root: Path) -> str:
    records = [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return canonical_sha256(records)


def _dataset_root(root: Path, directory_name: str) -> Path:
    direct = root / directory_name
    if direct.is_dir():
        return direct
    if root.name == directory_name and root.is_dir():
        return root
    raise FileNotFoundError(f"cannot find {directory_name} under {root}")


def _first_existing_dir(root: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = root if candidate == "." else root / candidate
        if path.is_dir():
            return path
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"none of the expected directories exist under {root}: {joined}")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} must contain a mapping")
    return payload


def _require_int_sequence(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an integer sequence")
    return tuple((_require_int(item, f"{field}[{index}]") for index, item in enumerate(value)))


def _read_space_table(path: Path, name: str) -> dict[int, str]:
    rows: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw_id, value = line.split(maxsplit=1)
            except ValueError as error:
                raise ValueError(f"{name}:{line_number} must contain '<id> <value>'") from error
            row_id = int(raw_id)
            if row_id in rows:
                raise ValueError(f"{name} contains duplicate id {row_id}")
            rows[row_id] = value
    return rows


def _first_present(row: Mapping[str, Any], keys: Sequence[str], *, field: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    raise ValueError(f"{field} is missing")


def _int_from_row(row: Mapping[str, Any], keys: Sequence[str]) -> int:
    value = _first_present(row, keys, field="/".join(keys))
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{'/'.join(keys)} must be an integer") from error
    return _require_int(value, "/".join(keys))


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _validate_portable_relative(value: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"non-portable ordered-record key: {value}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"non-portable ordered-record key: {value}")
