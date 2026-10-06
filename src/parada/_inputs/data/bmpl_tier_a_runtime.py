from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from parada._inputs.data.bmpl_input_registry import REFERENCE_DATA_SUBDIRS
from parada._inputs.data.bmpl_reference_manifests import DATASET_CLASS_COUNTS
from parada._inputs.data.bmpl_reference_records import (
    OrderedRecord,
    OrderedRecordSet,
    read_bmpl_ordered_records,
)
from parada._inputs.hashing import canonical_sha256, file_sha256
from parada._inputs.reference.class_names import resolve_official_class_names

TIER_A_DATASETS = ("cifar10", "cifar100", "spawrious", "stanforddogs", "food101", "artbench")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
_SPAWRIOUS_PACKAGE_COMMIT = "651e90415bfb414a5b6579cb1f53b83e6bb8c165"
_SPAWRIOUS_PACKAGE_CLASS_LIST = ("bulldog", "corgi", "dachshund", "labrador")
_SPAWRIOUS_BMPL_CLASS_NAMES = tuple(name.title() for name in _SPAWRIOUS_PACKAGE_CLASS_LIST)
_SPAWRIOUS_BMPL_CLASS_TO_LABEL = {
    name: index for index, name in enumerate(_SPAWRIOUS_PACKAGE_CLASS_LIST)
}
_ARTBENCH_CLASS_NAMES = (
    "Art Nouveau",
    "Baroque",
    "Expressionism",
    "Impressionism",
    "Post-Impressionism",
    "Realism",
    "Renaissance",
    "Romanticism",
    "Surrealism",
    "Ukiyo-e",
)
_STANFORDDOGS_CLASS_NAMES = (
    "Chihuaha",
    "Japanese Spaniel",
    "Maltese Dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim Spaniel",
    "Papillon",
    "Toy Terrier",
    "Rhodesian Ridgeback",
    "Afghan Hound",
    "Basset Hound",
    "Beagle",
    "Bloodhound",
    "Bluetick",
    "Black-and-tan Coonhound",
    "Walker Hound",
    "English Foxhound",
    "Redbone",
    "Borzoi",
    "Irish Wolfhound",
    "Italian Greyhound",
    "Whippet",
    "Ibizian Hound",
    "Norwegian Elkhound",
    "Otterhound",
    "Saluki",
    "Scottish Deerhound",
    "Weimaraner",
    "Staffordshire Bullterrier",
    "American Staffordshire Terrier",
    "Bedlington Terrier",
    "Border Terrier",
    "Kerry Blue Terrier",
    "Irish Terrier",
    "Norfolk Terrier",
    "Norwich Terrier",
    "Yorkshire Terrier",
    "Wirehaired Fox Terrier",
    "Lakeland Terrier",
    "Sealyham Terrier",
    "Airedale",
    "Cairn",
    "Australian Terrier",
    "Dandi Dinmont",
    "Boston Bull",
    "Miniature Schnauzer",
    "Giant Schnauzer",
    "Standard Schnauzer",
    "Scotch Terrier",
    "Tibetan Terrier",
    "Silky Terrier",
    "Soft-coated Wheaten Terrier",
    "West Highland White Terrier",
    "Lhasa",
    "Flat-coated Retriever",
    "Curly-coater Retriever",
    "Golden Retriever",
    "Labrador Retriever",
    "Chesapeake Bay Retriever",
    "German Short-haired Pointer",
    "Vizsla",
    "English Setter",
    "Irish Setter",
    "Gordon Setter",
    "Brittany",
    "Clumber",
    "English Springer Spaniel",
    "Welsh Springer Spaniel",
    "Cocker Spaniel",
    "Sussex Spaniel",
    "Irish Water Spaniel",
    "Kuvasz",
    "Schipperke",
    "Groenendael",
    "Malinois",
    "Briard",
    "Kelpie",
    "Komondor",
    "Old English Sheepdog",
    "Shetland Sheepdog",
    "Collie",
    "Border Collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German Shepard",
    "Doberman",
    "Miniature Pinscher",
    "Greater Swiss Mountain Dog",
    "Bernese Mountain Dog",
    "Appenzeller",
    "EntleBucher",
    "Boxer",
    "Bull Mastiff",
    "Tibetan Mastiff",
    "French Bulldog",
    "Great Dane",
    "Saint Bernard",
    "Eskimo Dog",
    "Malamute",
    "Siberian Husky",
    "Affenpinscher",
    "Basenji",
    "Pug",
    "Leonberg",
    "Newfoundland",
    "Great Pyrenees",
    "Samoyed",
    "Pomeranian",
    "Chow",
    "Keeshond",
    "Brabancon Griffon",
    "Pembroke",
    "Cardigan",
    "Toy Poodle",
    "Miniature Poodle",
    "Standard Poodle",
    "Mexican Hairless",
    "Dingo",
    "Dhole",
    "African Hunting Dog",
)


@dataclass(frozen=True, slots=True)
class TierASample:
    position: int
    sample_id: str
    label: int
    relative_path_or_native_index: str


@dataclass(frozen=True, slots=True)
class TierARuntimeManifest:
    dataset_id: str
    split: str
    builder_id: str
    class_names: tuple[str, ...]
    class_order_hash: str
    records: tuple[TierASample, ...]
    dataset_asset_identity: tuple[tuple[str, str], ...]
    manifest_hash: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "builder_id": self.builder_id,
            "class_names": self.class_names,
            "class_order_hash": self.class_order_hash,
            "records": [asdict(row) for row in self.records],
            "dataset_asset_identity": self.dataset_asset_identity,
        }

    def computed_hash(self) -> str:
        return canonical_sha256(self.payload_without_hash())


@dataclass(slots=True)
class TierAImageDataset:
    dataset_id: str
    split: str
    reference_data_root: Path
    records: tuple[TierASample, ...]
    transform: Callable[[Image.Image], Any] | None = None
    _cifar_store: Any = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, Any, int]:
        sample = self.records[index]
        if self.dataset_id in {"cifar10", "cifar100"}:
            if self._cifar_store is None:
                self._cifar_store = _CifarImageStore(
                    self.dataset_id, self.reference_data_root, self.split
                )
            image = self._cifar_store.load(sample.position)
        else:
            image = _load_single_image(
                self.dataset_id, self.reference_data_root, self.split, sample
            )
        value = self.transform(image) if self.transform is not None else image
        return (sample.sample_id, value, sample.label)


@dataclass(frozen=True, slots=True)
class TierADatasetBundle:
    dataset_id: str
    class_names: tuple[str, ...]
    support_dataset: TierAImageDataset
    test_dataset: TierAImageDataset
    support_records: tuple[TierASample, ...]
    test_records: tuple[TierASample, ...]
    support_manifest_hash: str
    test_manifest_hash: str
    class_order_hash: str


def load_tier_a_dataset(
    dataset_id: str,
    reference_data_root: str | Path,
    *,
    support_positions: Sequence[int],
    transform: Callable[[Image.Image], Any] | None = None,
) -> TierADatasetBundle:
    """Load a runner-facing Tier A bundle without materializing all images."""
    dataset_root = _resolve_dataset_root(reference_data_root, dataset_id)
    support_manifest = build_tier_a_train_runtime_manifest(
        dataset_id, dataset_root, selected_positions=support_positions
    )
    test_manifest = build_tier_a_test_runtime_manifest(dataset_id, dataset_root)
    support_dataset = TierAImageDataset(
        dataset_id=dataset_id,
        split="train",
        reference_data_root=dataset_root,
        records=support_manifest.records,
        transform=transform,
    )
    test_dataset = TierAImageDataset(
        dataset_id=dataset_id,
        split="test",
        reference_data_root=dataset_root,
        records=test_manifest.records,
        transform=transform,
    )
    return TierADatasetBundle(
        dataset_id=dataset_id,
        class_names=support_manifest.class_names,
        support_dataset=support_dataset,
        test_dataset=test_dataset,
        support_records=support_manifest.records,
        test_records=test_manifest.records,
        support_manifest_hash=support_manifest.manifest_hash,
        test_manifest_hash=test_manifest.manifest_hash,
        class_order_hash=support_manifest.class_order_hash,
    )


def build_tier_a_train_runtime_manifest(
    dataset_id: str,
    reference_data_root: str | Path,
    *,
    selected_positions: Sequence[int] | None = None,
) -> TierARuntimeManifest:
    """Build a portable train/support manifest from B0-compatible train order."""
    _require_tier_a_dataset(dataset_id)
    dataset_root = _resolve_dataset_root(reference_data_root, dataset_id)
    records = read_bmpl_ordered_records(dataset_id, dataset_root)
    _validate_class_count(dataset_id, records.records)
    positions = _normalize_positions(
        selected_positions if selected_positions is not None else range(len(records.records)),
        len(records.records),
    )
    rows = tuple(
        _sample_from_record(dataset_id, "train", position, records.records[position])
        for position in positions
    )
    return _runtime_manifest(
        dataset_id=dataset_id,
        split="train",
        builder_id=f"bmpl-tier-a-{dataset_id}-train-support-runtime-v1",
        class_names=resolve_tier_a_class_names(dataset_id, dataset_root),
        rows=rows,
        asset_identity=records.dataset_asset_identity,
    )


def build_tier_a_test_runtime_manifest(
    dataset_id: str, reference_data_root: str | Path
) -> TierARuntimeManifest:
    """Build a portable complete-test manifest in fedprototype-compatible order."""
    _require_tier_a_dataset(dataset_id)
    dataset_root = _resolve_dataset_root(reference_data_root, dataset_id)
    records = _read_test_records(dataset_id, dataset_root)
    _validate_class_count(dataset_id, records.records)
    rows = tuple(
        (
            _sample_from_record(dataset_id, "test", position, record)
            for position, record in enumerate(records.records)
        )
    )
    return _runtime_manifest(
        dataset_id=dataset_id,
        split="test",
        builder_id=f"bmpl-tier-a-{dataset_id}-test-runtime-v1",
        class_names=resolve_tier_a_class_names(dataset_id, dataset_root),
        rows=rows,
        asset_identity=records.dataset_asset_identity,
    )


def resolve_tier_a_class_names(
    dataset_id: str, reference_data_root: str | Path | None = None
) -> tuple[str, ...]:
    """Resolve class names in the label order used by the runtime manifests."""
    _require_tier_a_dataset(dataset_id)
    if dataset_id in {"cifar10", "cifar100"}:
        return tuple(resolve_official_class_names(dataset_id))
    if dataset_id == "spawrious":
        return _SPAWRIOUS_BMPL_CLASS_NAMES
    if dataset_id == "stanforddogs":
        return _STANFORDDOGS_CLASS_NAMES
    if dataset_id == "artbench":
        return _ARTBENCH_CLASS_NAMES
    if dataset_id == "food101":
        if reference_data_root is None:
            raise ValueError("food101 class names require a local dataset root")
        base = _first_existing_dir(Path(reference_data_root), ("food-101", "."))
        train_meta = _require_file(base / "meta" / "train.json")
        test_meta = _require_file(base / "meta" / "test.json")
        names = sorted(
            {
                *json.loads(train_meta.read_text(encoding="utf-8")).keys(),
                *json.loads(test_meta.read_text(encoding="utf-8")).keys(),
            }
        )
        return tuple(str(name) for name in names)
    raise AssertionError(f"unreachable dataset: {dataset_id}")


def validate_tier_a_runtime_manifest(manifest: TierARuntimeManifest) -> TierARuntimeManifest:
    _require_tier_a_dataset(manifest.dataset_id)
    if manifest.split not in {"train", "test"}:
        raise ValueError("Tier A runtime manifest split must be train or test")
    if not manifest.builder_id:
        raise ValueError("Tier A runtime manifest builder_id must be nonempty")
    if manifest.class_order_hash != canonical_sha256(manifest.class_names):
        raise ValueError("Tier A runtime class_order_hash changed")
    expected_count = DATASET_CLASS_COUNTS[manifest.dataset_id]
    if len(manifest.class_names) != expected_count:
        raise ValueError(
            f"{manifest.dataset_id} class count changed: expected "
            f"{expected_count}, found {len(manifest.class_names)}"
        )
    if not manifest.records:
        raise ValueError("Tier A runtime manifest must contain at least one row")
    seen: set[str] = set()
    for row in manifest.records:
        _validate_sample(manifest.dataset_id, manifest.split, row)
        if row.sample_id in seen:
            raise ValueError(f"duplicate Tier A sample_id: {row.sample_id}")
        seen.add(row.sample_id)
    if not manifest.dataset_asset_identity:
        raise ValueError("Tier A runtime manifest requires dataset_asset_identity")
    if manifest.manifest_hash != manifest.computed_hash():
        raise ValueError("Tier A runtime manifest_hash does not match content")
    return manifest


def _read_test_records(dataset_id: str, root: Path) -> OrderedRecordSet:
    if dataset_id == "cifar10":
        return _read_cifar_test_records(
            dataset_id, root, batch_name="test_batch", label_key="labels"
        )
    if dataset_id == "cifar100":
        return _read_cifar_test_records(
            dataset_id, root, batch_name="test", label_key="fine_labels"
        )
    if dataset_id == "food101":
        return _read_food101_test_records(root)
    if dataset_id == "artbench":
        return _read_artbench_test_records(root)
    if dataset_id == "stanforddogs":
        return _read_stanforddogs_test_records(root)
    if dataset_id == "spawrious":
        return _read_spawrious_test_records(root)
    raise ValueError(f"unsupported Tier A test dataset: {dataset_id}")


def _read_cifar_test_records(
    dataset_id: str, root: Path, *, batch_name: str, label_key: str
) -> OrderedRecordSet:
    directory = "cifar-10-batches-py" if dataset_id == "cifar10" else "cifar-100-python"
    data_root = _dataset_root(root, directory)
    batch_path = _require_file(data_root / batch_name)
    payload = _load_pickle(batch_path)
    labels = _require_int_sequence(payload.get(label_key), f"{batch_name}.{label_key}")
    return _record_set(
        dataset_id=dataset_id,
        split="test",
        reader_id=f"bmpl-tier-a-{dataset_id}-test-torchvision-pickle-v1",
        records=[
            OrderedRecord(f"native_index/{position:08d}", label)
            for position, label in enumerate(labels)
        ],
        asset_identity=_identity(
            reader_id=f"bmpl-tier-a-{dataset_id}-test-torchvision-pickle-v1",
            source_files=[batch_path],
            extra={"label_order_sha256": canonical_sha256(labels)},
        ),
    )


def _read_food101_test_records(root: Path) -> OrderedRecordSet:
    base = _first_existing_dir(root, ("food-101", "."))
    meta_path = _require_file(base / "meta" / "test.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("Food101 test.json must contain a mapping")
    classes = tuple(sorted(str(name) for name in metadata))
    class_to_label = {name: index for index, name in enumerate(classes)}
    rows: list[OrderedRecord] = []
    for class_name, raw_paths in metadata.items():
        if not isinstance(raw_paths, list):
            raise ValueError(f"Food101 test.json class {class_name!r} must contain a list")
        for raw_path in raw_paths:
            rows.append(
                OrderedRecord(f"images/{str(raw_path)}.jpg", class_to_label[str(class_name)])
            )
    return _record_set(
        dataset_id="food101",
        split="test",
        reader_id="bmpl-tier-a-food101-test-torchvision-json-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-tier-a-food101-test-torchvision-json-v1",
            source_files=[meta_path],
            extra={"class_order_sha256": canonical_sha256(classes)},
        ),
    )


def _read_artbench_test_records(root: Path) -> OrderedRecordSet:
    test_root = _first_existing_dir(root, ("artbench-10/test", "ArtBench-10/test", "test"))
    rows = _imagefolder_records(test_root)
    return _record_set(
        dataset_id="artbench",
        split="test",
        reader_id="bmpl-tier-a-artbench-test-imagefolder-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-tier-a-artbench-test-imagefolder-v1",
            extra={"imagefolder_inventory_sha256": _directory_inventory_hash(test_root)},
        ),
    )


def _read_stanforddogs_test_records(root: Path) -> OrderedRecordSet:
    base = _first_existing_dir(root, ("StanfordDogs", "."))
    split_file = _require_file(base / "test_list.mat")
    try:
        loadmat = import_module("scipy.io").loadmat
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "StanfordDogs test order requires scipy.io.loadmat for test_list.mat"
        ) from error
    payload = loadmat(split_file, squeeze_me=True)
    if "annotation_list" not in payload or "labels" not in payload:
        raise ValueError("StanfordDogs test_list.mat lacks annotation_list/labels")
    names = _mat_vector(payload["annotation_list"], field="annotation_list")
    labels = _mat_vector(payload["labels"], field="labels")
    if len(names) != len(labels):
        raise ValueError("StanfordDogs annotation_list and labels lengths differ")
    return _record_set(
        dataset_id="stanforddogs",
        split="test",
        reader_id="bmpl-tier-a-stanforddogs-test-list-mat-v1",
        records=[
            OrderedRecord(f"{str(name)}.jpg", int(label) - 1)
            for name, label in zip(names, labels, strict=True)
        ],
        asset_identity=_identity(
            reader_id="bmpl-tier-a-stanforddogs-test-list-mat-v1", source_files=[split_file]
        ),
    )


def _read_spawrious_test_records(root: Path) -> OrderedRecordSet:
    data_root = _first_existing_dir(root, ("spawrious224", "."))
    test_locations = {
        "bulldog": ("mountain", "mountain"),
        "dachshund": ("snow", "snow"),
        "labrador": ("desert", "desert"),
        "corgi": ("jungle", "jungle"),
    }
    rows: list[OrderedRecord] = []
    ordered_directory_entries: dict[str, list[str]] = {}
    for phase in (0, 1):
        for class_name in ("bulldog", "dachshund", "labrador", "corgi"):
            rows.extend(
                _spawrious_directory_records(
                    data_root,
                    phase=phase,
                    location=test_locations[class_name][phase],
                    class_name=class_name,
                    label=_SPAWRIOUS_BMPL_CLASS_TO_LABEL[class_name],
                    ordered_directory_entries=ordered_directory_entries,
                )
            )
    return _record_set(
        dataset_id="spawrious",
        split="test",
        reader_id="bmpl-tier-a-spawrious-test-o2o-hard-os-listdir-v1",
        records=rows,
        asset_identity=_identity(
            reader_id="bmpl-tier-a-spawrious-test-o2o-hard-os-listdir-v1",
            extra={
                "benchmark": "o2o_hard",
                "spawrious_package_commit": _SPAWRIOUS_PACKAGE_COMMIT,
                "spawrious_class_list_sha256": canonical_sha256(_SPAWRIOUS_PACKAGE_CLASS_LIST),
                "ordered_directory_entries_sha256": canonical_sha256(ordered_directory_entries),
            },
        ),
    )


def _load_single_image(
    dataset_id: str, dataset_root: Path, split: str, sample: TierASample
) -> Image.Image:
    if dataset_id in {"cifar10", "cifar100"}:
        return _CifarImageStore(dataset_id, dataset_root, split).load(sample.position)
    with Image.open(_path_image_path(dataset_id, dataset_root, split, sample)) as image:
        return image.convert("RGB")


def _path_image_path(dataset_id: str, root: Path, split: str, sample: TierASample) -> Path:
    relative = sample.relative_path_or_native_index
    if dataset_id == "stanforddogs":
        base = _first_existing_dir(root, ("StanfordDogs", "."))
        return _require_file(base / "Images" / relative)
    if dataset_id == "food101":
        base = _first_existing_dir(root, ("food-101", "."))
        return _require_file(base / relative)
    if dataset_id == "artbench":
        split_root = _first_existing_dir(
            root, (f"artbench-10/{split}", f"ArtBench-10/{split}", split)
        )
        return _require_file(split_root / relative)
    if dataset_id == "spawrious":
        data_root = _first_existing_dir(root, ("spawrious224", "."))
        return _require_file(data_root / relative)
    raise ValueError(f"unsupported path-image dataset: {dataset_id}")


class _CifarImageStore:
    def __init__(self, dataset_id: str, root: Path, split: str) -> None:
        if dataset_id == "cifar10":
            directory, batches, label_key = (
                "cifar-10-batches-py",
                [f"data_batch_{index}" for index in range(1, 6)]
                if split == "train"
                else ["test_batch"],
                "labels",
            )
        elif dataset_id == "cifar100":
            directory, batches, label_key = (
                "cifar-100-python",
                ["train"] if split == "train" else ["test"],
                "fine_labels",
            )
        else:
            raise ValueError(f"unsupported CIFAR dataset: {dataset_id}")
        self._images: list[np.ndarray] = []
        self._labels: list[int] = []
        data_root = _dataset_root(root, directory)
        for batch in batches:
            payload = _load_pickle(_require_file(data_root / batch))
            raw_data = np.asarray(payload.get("data"))
            if raw_data.ndim != 2 or raw_data.shape[1] != 3072:
                raise ValueError(f"{dataset_id} {batch}.data must have shape [N, 3072]")
            labels = _require_int_sequence(payload.get(label_key), f"{batch}.{label_key}")
            if raw_data.shape[0] != len(labels):
                raise ValueError(f"{dataset_id} {batch} data/label lengths differ")
            self._images.extend(raw_data)
            self._labels.extend(labels)

    def load(self, position: int) -> Image.Image:
        if position < 0 or position >= len(self._images):
            raise IndexError(position)
        array = (
            np.asarray(self._images[position], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
        )
        return Image.fromarray(array, mode="RGB")


def _sample_from_record(
    dataset_id: str, split: str, position: int, record: OrderedRecord
) -> TierASample:
    relative = str(record.relative_path_or_native_index)
    label = int(record.label)
    return TierASample(
        position=position,
        sample_id=_sample_id(dataset_id, split, position, relative, label),
        label=label,
        relative_path_or_native_index=relative,
    )


def _sample_id(dataset_id: str, split: str, position: int, relative: str, label: int) -> str:
    _ = label
    suffix = canonical_sha256(
        {
            "dataset_id": dataset_id,
            "split": split,
            "position": position,
            "relative_path_or_native_index": relative,
        }
    )[:16]
    return f"{dataset_id}/{split}/{position:08d}/{suffix}"


def _runtime_manifest(
    *,
    dataset_id: str,
    split: str,
    builder_id: str,
    class_names: Sequence[str],
    rows: Sequence[TierASample],
    asset_identity: Iterable[tuple[str, str]],
) -> TierARuntimeManifest:
    manifest = TierARuntimeManifest(
        dataset_id=dataset_id,
        split=split,
        builder_id=builder_id,
        class_names=tuple(str(name) for name in class_names),
        class_order_hash=canonical_sha256(tuple(str(name) for name in class_names)),
        records=tuple(rows),
        dataset_asset_identity=tuple(
            sorted(((str(key), str(value)) for key, value in asset_identity))
        ),
        manifest_hash="",
    )
    return validate_tier_a_runtime_manifest(
        TierARuntimeManifest(
            dataset_id=manifest.dataset_id,
            split=manifest.split,
            builder_id=manifest.builder_id,
            class_names=manifest.class_names,
            class_order_hash=manifest.class_order_hash,
            records=manifest.records,
            dataset_asset_identity=manifest.dataset_asset_identity,
            manifest_hash=manifest.computed_hash(),
        )
    )


def _record_set(
    *,
    dataset_id: str,
    split: str,
    reader_id: str,
    records: Iterable[OrderedRecord],
    asset_identity: tuple[tuple[str, str], ...],
) -> OrderedRecordSet:
    class_count = DATASET_CLASS_COUNTS[dataset_id]
    rows: list[OrderedRecord] = []
    seen: set[str] = set()
    for position, row in enumerate(records):
        relative = str(row.relative_path_or_native_index)
        _validate_portable_relative(relative)
        if relative in seen:
            raise ValueError(f"duplicate ordered-record key: {relative}")
        seen.add(relative)
        label = _require_int(row.label, f"label at position {position}")
        if not 0 <= label < class_count:
            raise ValueError(f"label at position {position} is outside [0, {class_count}): {label}")
        rows.append(OrderedRecord(relative, label))
    if not rows:
        raise ValueError(f"{dataset_id} {split} reader produced no records")
    return OrderedRecordSet(
        dataset_id=dataset_id,
        split=split,
        reader_id=reader_id,
        records=tuple(rows),
        dataset_asset_identity=tuple(sorted(asset_identity)),
    )


def _validate_sample(dataset_id: str, split: str, row: TierASample) -> None:
    if row.position < 0:
        raise ValueError("Tier A sample position must be nonnegative")
    if not row.sample_id.startswith(f"{dataset_id}/{split}/{row.position:08d}/"):
        raise ValueError(f"Tier A sample_id does not match dataset/split/position: {row.sample_id}")
    if row.sample_id != _sample_id(
        dataset_id, split, row.position, row.relative_path_or_native_index, row.label
    ):
        raise ValueError("Tier A sample_id hash does not match row content")
    if not 0 <= row.label < DATASET_CLASS_COUNTS[dataset_id]:
        raise ValueError(f"Tier A sample label is outside class range: {row.label}")
    _validate_portable_relative(row.relative_path_or_native_index)


def _resolve_dataset_root(reference_data_root: str | Path, dataset_id: str) -> Path:
    root = Path(reference_data_root)
    candidates = [root]
    subdir = REFERENCE_DATA_SUBDIRS.get(dataset_id)
    if subdir is not None:
        candidates.insert(0, root / subdir)
    for candidate in candidates:
        if _looks_like_dataset_root(dataset_id, candidate):
            return candidate
    if root.is_dir():
        return root
    raise FileNotFoundError(root)


def _looks_like_dataset_root(dataset_id: str, root: Path) -> bool:
    if not root.is_dir():
        return False
    markers = {
        "cifar10": ("cifar-10-batches-py",),
        "cifar100": ("cifar-100-python",),
        "food101": ("food-101/meta", "meta"),
        "artbench": ("artbench-10/train", "ArtBench-10/train", "train"),
        "stanforddogs": ("StanfordDogs/train_list.mat", "train_list.mat"),
        "spawrious": ("spawrious224", "0"),
    }[dataset_id]
    return any((root / marker).exists() for marker in markers)


def _normalize_positions(positions: Iterable[int], row_count: int) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw in positions:
        position = _require_int(raw, "selected position")
        if not 0 <= position < row_count:
            raise ValueError(f"selected train position is outside [0, {row_count}): {position}")
        normalized.append(position)
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected train positions contain duplicates")
    return tuple(normalized)


def _validate_class_count(dataset_id: str, records: Sequence[OrderedRecord]) -> None:
    labels = {row.label for row in records}
    if labels and (min(labels) < 0 or max(labels) >= DATASET_CLASS_COUNTS[dataset_id]):
        raise ValueError(f"{dataset_id} labels exceed frozen class count")


def _imagefolder_records(root: Path) -> tuple[OrderedRecord, ...]:
    class_names = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
    if not class_names:
        raise FileNotFoundError(f"no class folders found under {root}")
    rows: list[OrderedRecord] = []
    for label, class_name in enumerate(class_names):
        class_root = root / class_name
        for path in sorted(item for item in class_root.rglob("*") if item.is_file()):
            if path.suffix.lower() in _IMAGE_EXTENSIONS:
                rows.append(OrderedRecord(path.relative_to(root).as_posix(), label))
    return tuple(rows)


def _spawrious_directory_records(
    data_root: Path,
    *,
    phase: int,
    location: str,
    class_name: str,
    label: int,
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
        for name in ordered_names
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


def _require_tier_a_dataset(dataset_id: str) -> None:
    if dataset_id not in TIER_A_DATASETS:
        raise ValueError(f"unsupported BMPL Tier A dataset: {dataset_id}")
