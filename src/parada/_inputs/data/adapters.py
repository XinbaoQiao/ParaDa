from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, DTD, OxfordIIITPet

from parada._inputs.benchmarks.schema import SampleEntry
from parada._inputs.hashing import canonical_sha256, file_sha256


@dataclass(slots=True)
class DatasetView:
    dataset_id: str
    version: str
    class_names: tuple[str, ...]
    splits: dict[str, Any]
    records: dict[str, tuple[SampleEntry, ...]]
    raw_data_fingerprint: str


class DatasetAdapter(Protocol):
    dataset_id: str

    def load(self) -> DatasetView: ...


class StableIdDataset(Dataset[tuple[str, Any, int]]):
    def __init__(self, dataset: Any, entries: tuple[SampleEntry, ...]) -> None:
        if len(dataset) != len(entries):
            raise ValueError("dataset and stable-ID entries have different lengths")
        self.dataset = dataset
        self.entries = entries

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[str, Any, int]:
        image, label = self.dataset[index]
        entry = self.entries[index]
        if int(label) != entry.label:
            raise RuntimeError(f"dataset label drift for sample {entry.sample_id}")
        return (entry.sample_id, image, entry.label)


class EmptyDataset(Dataset[Any]):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:
        raise IndexError(index)


def _fingerprint_trees(dataset_id: str, base: Path, roots: tuple[Path, ...]) -> str:
    """Hash extracted dataset trees using portable paths relative to one root."""
    files = sorted(path for root in roots for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"{dataset_id} raw files are unavailable under {base}")
    return canonical_sha256(
        [
            {
                "name": path.relative_to(base).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ]
    )


def _fingerprint_tree(dataset_id: str, base: Path) -> str:
    return _fingerprint_trees(dataset_id, base, (base,))


def _image_file_entries(
    dataset: Any,
    *,
    public_split: str,
    source_split: str,
    image_files_attribute: str,
    images_root_attribute: str,
    split_source_prefix: str,
) -> tuple[SampleEntry, ...]:
    image_files = tuple(Path(path) for path in getattr(dataset, image_files_attribute))
    labels = tuple(int(label) for label in dataset._labels)
    images_root = Path(getattr(dataset, images_root_attribute))
    if len(image_files) != len(labels) or len(image_files) != len(dataset):
        raise RuntimeError(f"{split_source_prefix} image/label cardinality mismatch")
    entries: list[SampleEntry] = []
    for image_file, label in zip(image_files, labels, strict=True):
        try:
            relative_image = image_file.relative_to(images_root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"{split_source_prefix} image is outside its declared image root: {image_file}"
            ) from error
        entries.append(
            SampleEntry(
                sample_id=f"{public_split}/{relative_image}",
                label=label,
                split_source=f"{split_source_prefix}:{source_split}",
            )
        )
    return tuple(entries)


class _CIFARAdapter:
    dataset_id: str
    dataset_class: type[Any]
    version: str
    archive_directory: str
    split_source: str

    def __init__(self, root: str | Path, *, transform: Any = None, download: bool = False) -> None:
        self.root = Path(root)
        self.transform = transform
        self.download = download

    def _entries(self, dataset: Any, split: str) -> tuple[SampleEntry, ...]:
        return tuple(
            (
                SampleEntry(
                    sample_id=f"{split}/{index:05d}",
                    label=int(label),
                    split_source=self.split_source,
                )
                for index, label in enumerate(dataset.targets)
            )
        )

    def _fingerprint(self) -> str:
        base = self.root / self.archive_directory
        return _fingerprint_tree(self.dataset_id, base)

    def load(self) -> DatasetView:
        train = self.dataset_class(
            self.root, train=True, transform=self.transform, download=self.download
        )
        test = self.dataset_class(
            self.root, train=False, transform=self.transform, download=self.download
        )
        class_names = tuple(str(name) for name in train.classes)
        if class_names != tuple(str(name) for name in test.classes):
            raise RuntimeError(f"{self.dataset_id} train/test class order mismatch")
        train_entries = self._entries(train, "train")
        test_entries = self._entries(test, "test")
        return DatasetView(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=class_names,
            splits={
                "train": StableIdDataset(train, train_entries),
                "val": EmptyDataset(),
                "test": StableIdDataset(test, test_entries),
            },
            records={"train": train_entries, "val": (), "test": test_entries},
            raw_data_fingerprint=self._fingerprint(),
        )


class CIFAR10Adapter(_CIFARAdapter):
    dataset_id = "cifar10"
    dataset_class = CIFAR10
    version = "torchvision-cifar10-python"
    archive_directory = "cifar-10-batches-py"
    split_source = "torchvision:CIFAR10:official"


class CIFAR100Adapter(_CIFARAdapter):
    """Torchvision CIFAR-100 using the official fine-label class order."""

    dataset_id = "cifar100"
    dataset_class = CIFAR100
    version = "torchvision-cifar100-python"
    archive_directory = "cifar-100-python"
    split_source = "torchvision:CIFAR100:official"


class DTDAdapter:
    """DTD partition 1 with the author-code train+val versus test protocol."""

    dataset_id = "dtd"
    dataset_class: type[Any] = DTD
    version = "torchvision-dtd-r1.0.1-partition1"
    class_count = 47
    partition = 1
    split_source_prefix = "torchvision:DTD:partition1"

    def __init__(self, root: str | Path, *, transform: Any = None, download: bool = False) -> None:
        self.root = Path(root)
        self.transform = transform
        self.download = download

    def _load_split(self, split: str) -> Any:
        return self.dataset_class(
            self.root,
            split=split,
            partition=self.partition,
            transform=self.transform,
            download=self.download,
        )

    def _entries(
        self, dataset: Any, *, public_split: str, source_split: str
    ) -> tuple[SampleEntry, ...]:
        return _image_file_entries(
            dataset,
            public_split=public_split,
            source_split=source_split,
            image_files_attribute="_image_files",
            images_root_attribute="_images_folder",
            split_source_prefix=self.split_source_prefix,
        )

    def load(self) -> DatasetView:
        train = self._load_split("train")
        val = self._load_split("val")
        test = self._load_split("test")
        class_names = tuple(str(name) for name in train.classes)
        if len(class_names) != self.class_count:
            raise RuntimeError(
                f"dtd class count mismatch: expected {self.class_count}, found {len(class_names)}"
            )
        if any(tuple(str(name) for name in split.classes) != class_names for split in (val, test)):
            raise RuntimeError("dtd train/val/test class order mismatch")
        train_entries = self._entries(train, public_split="train", source_split="train")
        val_entries = self._entries(val, public_split="train", source_split="val")
        test_entries = self._entries(test, public_split="test", source_split="test")
        combined_train = ConcatDataset((train, val))
        combined_entries = train_entries + val_entries
        data_root = Path(train._data_folder)
        return DatasetView(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=class_names,
            splits={
                "train": StableIdDataset(combined_train, combined_entries),
                "val": EmptyDataset(),
                "test": StableIdDataset(test, test_entries),
            },
            records={"train": combined_entries, "val": (), "test": test_entries},
            raw_data_fingerprint=_fingerprint_tree(self.dataset_id, data_root),
        )


class OxfordIIITPetAdapter:
    """Oxford-IIIT Pets closed-set trainval/test category protocol."""

    dataset_id = "oxford_pets"
    dataset_class: type[Any] = OxfordIIITPet
    version = "torchvision-oxford-iiit-pet"
    class_count = 37
    split_source_prefix = "torchvision:OxfordIIITPet"

    def __init__(self, root: str | Path, *, transform: Any = None, download: bool = False) -> None:
        self.root = Path(root)
        self.transform = transform
        self.download = download

    def _load_split(self, split: str) -> Any:
        return self.dataset_class(
            self.root,
            split=split,
            target_types="category",
            transform=self.transform,
            download=self.download,
        )

    def _entries(
        self, dataset: Any, *, public_split: str, source_split: str
    ) -> tuple[SampleEntry, ...]:
        return _image_file_entries(
            dataset,
            public_split=public_split,
            source_split=source_split,
            image_files_attribute="_images",
            images_root_attribute="_images_folder",
            split_source_prefix=self.split_source_prefix,
        )

    def load(self) -> DatasetView:
        train = self._load_split("trainval")
        test = self._load_split("test")
        class_names = tuple(str(name) for name in train.classes)
        if len(class_names) != self.class_count:
            raise RuntimeError(
                f"oxford_pets class count mismatch: expected "
                f"{self.class_count}, found {len(class_names)}"
            )
        if tuple(str(name) for name in test.classes) != class_names:
            raise RuntimeError("oxford_pets trainval/test class order mismatch")
        train_entries = self._entries(train, public_split="train", source_split="trainval")
        test_entries = self._entries(test, public_split="test", source_split="test")
        base_folder = Path(train._base_folder)
        images_folder = Path(train._images_folder)
        annotations_folder = Path(train._anns_folder)
        return DatasetView(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=class_names,
            splits={
                "train": StableIdDataset(train, train_entries),
                "val": EmptyDataset(),
                "test": StableIdDataset(test, test_entries),
            },
            records={"train": train_entries, "val": (), "test": test_entries},
            raw_data_fingerprint=_fingerprint_trees(
                self.dataset_id, base_folder, (images_folder, annotations_folder)
            ),
        )


_DATASET_ADAPTER_CLASSES: dict[str, type[Any]] = {
    "cifar10": CIFAR10Adapter,
    "cifar100": CIFAR100Adapter,
    "dtd": DTDAdapter,
    "oxford_pets": OxfordIIITPetAdapter,
}


def create_dataset_adapter(
    dataset_id: str, root: str | Path, *, transform: Any = None, download: bool = False
) -> DatasetAdapter:
    """Construct an implemented adapter through the canonical dataset registry."""
    from parada._inputs.data.registry import get_dataset_spec

    spec = get_dataset_spec(dataset_id)
    try:
        adapter_class = _DATASET_ADAPTER_CLASSES[dataset_id]
    except KeyError as error:
        raise NotImplementedError(
            f"dataset adapter is not implemented: {dataset_id} ({spec.adapter_id})"
        ) from error
    return adapter_class(root, transform=transform, download=download)
