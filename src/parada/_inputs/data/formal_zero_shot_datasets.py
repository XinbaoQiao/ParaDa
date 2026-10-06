from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import GTSRB

from parada._inputs.benchmarks.schema import SampleEntry
from parada._inputs.data.adapters import (
    DatasetView,
    EmptyDataset,
    StableIdDataset,
    create_dataset_adapter,
)
from parada._inputs.hashing import canonical_sha256, file_sha256
from parada._inputs.reference.class_names import (
    OFFICIAL_DTD_CLASS_NAMES,
    OFFICIAL_OXFORD_PETS_CLASS_NAMES,
)

FORMAL_ZERO_SHOT_DATASET_IDS = (
    "cifar10",
    "cifar100",
    "stanforddogs",
    "oxford_pets",
    "cub2011",
    "flowers102",
    "food101",
    "places365",
    "resisc45",
    "eurosat",
    "dtd",
    "artbench",
    "waterbirds",
    "spawrious",
    "ham10000",
    "camelyon17",
    "oct2017",
    "plantvillage",
    "gtsrb",
)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
_HAM_CODES = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
_HAM_PROMPTS = (
    "Melanocytic nevi (benign moles)",
    "Melanoma (malignant skin cancer)",
    "Benign keratosis-like lesions (solar lentigines, seborrheic keratoses)",
    "Basal cell carcinoma",
    "Actinic keratoses / intraepithelial carcinoma (pre-cancerous lesions)",
    "Vascular lesions (angiomas, etc.)",
    "Dermatofibroma (benign skin growth)",
)
_CAMELYON_PROMPTS = ("normal lymph-node tissue", "metastatic tumor tissue")
_OCT_CLASSES = ("CNV", "DME", "DRUSEN", "NORMAL")
_OCT_PROMPTS = ("choroidal neovascularization", "diabetic macular edema", "drusen", "normal retina")
_EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
_RESISC45_CLASSES = (
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland",
)
_RESISC45_PROMPTS = tuple(f"{name.replace('_', ' ')} (aerial photo)" for name in _RESISC45_CLASSES)
_EUROSAT_PROMPTS = (
    "annual crop land  (aerial photo)",
    "forest  (aerial photo)",
    "brushland or shrubland  (aerial photo)",
    "highway or road  (aerial photo)",
    "industrial buildings or commercial buildings  (aerial photo)",
    "pasture land  (aerial photo)",
    "permanent crop land  (aerial photo)",
    "residential buildings or homes or apartments  (aerial photo)",
    "river  (aerial photo)",
    "lake or sea  (aerial photo)",
)
_PLANTVILLAGE_CLASSES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)
_PLANTVILLAGE_PROMPTS = tuple(
    name.replace("___", " disease: ").replace("_", " ") for name in _PLANTVILLAGE_CLASSES
)
_GTSRB_PROMPTS = tuple(f"German traffic sign class {index}" for index in range(43))


class _RecordDataset(Dataset[tuple[Any, int]]):
    """Small image-file dataset; absolute paths remain runtime-only."""

    def __init__(
        self, paths: Sequence[Path], entries: Sequence[SampleEntry], transform: Any
    ) -> None:
        if len(paths) != len(entries):
            raise ValueError("image paths and records have different lengths")
        self._paths = tuple(paths)
        self._entries = tuple(entries)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        with Image.open(self._paths[index]) as image:
            value = image.convert("RGB")
        if self._transform is not None:
            value = self._transform(value)
        return (value, self._entries[index].label)


class FormalDatasetView(DatasetView):
    """DatasetView with the fields required by the formal zero-shot runner."""

    __slots__ = (
        "canonical_class_names",
        "prompt_class_names",
        "group_metadata",
        "split_policy",
        "raw_data_binding",
        "evaluation_split",
    )

    def __init__(
        self,
        *,
        dataset_id: str,
        version: str,
        class_names: tuple[str, ...],
        prompt_class_names: tuple[str, ...],
        splits: dict[str, Any],
        records: dict[str, tuple[SampleEntry, ...]],
        group_metadata: Mapping[str, str],
        split_policy: Mapping[str, Any],
        raw_data_binding: Mapping[str, Any],
        evaluation_split: str,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            version=version,
            class_names=class_names,
            splits=splits,
            records=records,
            raw_data_fingerprint=str(raw_data_binding["fingerprint"]),
        )
        self.canonical_class_names = class_names
        self.prompt_class_names = prompt_class_names
        self.group_metadata = dict(group_metadata)
        self.split_policy = dict(split_policy)
        self.raw_data_binding = dict(raw_data_binding)
        self.evaluation_split = evaluation_split


def _normal(value: Any) -> str:
    return re.sub("[^a-z0-9]+", "", str(value).strip().lower())


def _approximate_mode(population: np.ndarray, draws: int, rng: np.random.RandomState) -> np.ndarray:
    """Local equivalent of sklearn's private ``_approximate_mode`` helper."""
    continuous = population * draws / population.sum()
    floored = np.floor(continuous)
    need = int(draws - floored.sum())
    if need:
        remainder = continuous - floored
        for value in np.sort(np.unique(remainder))[::-1]:
            indices = np.flatnonzero(remainder == value)
            amount = min(len(indices), need)
            if amount:
                chosen = rng.choice(indices, size=amount, replace=False)
                floored[chosen] += 1
                need -= amount
                if not need:
                    break
    return floored.astype(np.int64)


def _sklearn_stratified_split(
    labels: Sequence[int], *, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``train_test_split(..., stratify=labels)`` without sklearn.

    This is the one-split ``StratifiedShuffleSplit`` algorithm used by sklearn:
    the local implementation avoids making scikit-learn a runtime dependency
    while retaining its exact random ordering and class allocation semantics.
    """
    target = np.asarray(labels)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("stratified labels must be a non-empty one-dimensional sequence")
    classes, encoded, class_counts = np.unique(target, return_inverse=True, return_counts=True)
    del classes
    if int(class_counts.min()) < 2:
        raise ValueError("the least populated class must contain at least two samples")
    n_test = int(math.ceil(target.size * test_size))
    n_train = int(target.size - n_test)
    if n_test < len(class_counts) or n_train < len(class_counts):
        raise ValueError("train/test size must be at least the number of classes")
    rng = np.random.RandomState(random_state)
    class_indices = np.split(np.argsort(encoded, kind="mergesort"), np.cumsum(class_counts)[:-1])
    train_counts = _approximate_mode(class_counts, n_train, rng)
    test_counts = _approximate_mode(class_counts - train_counts, n_test, rng)
    train: list[int] = []
    test: list[int] = []
    for indices, train_count, test_count, class_count in zip(
        class_indices, train_counts, test_counts, class_counts, strict=True
    ):
        permutation = rng.permutation(int(class_count))
        train.extend(indices[permutation[:train_count]].tolist())
        test.extend(indices[permutation[train_count : train_count + test_count]].tolist())
    return (rng.permutation(train), rng.permutation(test))


def _images(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"image path is outside runtime data root: {path}") from error


def _find_image(
    root: Path, value: str, image_index: Mapping[str, Sequence[Path]] | None = None
) -> tuple[Path, str]:
    raw = Path(str(value).replace("\\", "/"))
    candidates = [raw] if raw.is_absolute() else [root / raw, root / "images" / raw]
    for candidate in candidates:
        if candidate.is_file():
            return (candidate, _relative(root, candidate))
    basename = raw.name
    matches = (
        tuple(image_index.get(basename, ()))
        if image_index is not None
        else tuple(path for path in _images(root) if path.name == basename)
    )
    if len(matches) == 1:
        return (matches[0], _relative(root, matches[0]))
    if len(matches) > 1 and raw.parent != Path("."):
        suffix_matches = tuple(path for path in matches if path.as_posix().endswith(raw.as_posix()))
        if len(suffix_matches) == 1:
            return (suffix_matches[0], _relative(root, suffix_matches[0]))
    if len(matches) > 1:
        raise ValueError(f"image basename is ambiguous below the runtime root: {value!r}")
    raise FileNotFoundError(f"cannot resolve image {value!r} below the runtime root")


def _fingerprint(
    root: Path, metadata_files: Iterable[Path] = (), data_roots: Iterable[Path] | None = None
) -> tuple[str, int]:
    inventory_roots = tuple(data_roots) if data_roots is not None else (root,)
    files = {
        path
        for inventory_root in inventory_roots
        for path in inventory_root.rglob("*")
        if path.is_file() and (data_roots is not None or path.suffix.lower() in _IMAGE_SUFFIXES)
    }
    metadata_set = {path for path in metadata_files if path.is_file()}
    files.update(metadata_set)
    if not files:
        raise FileNotFoundError(f"formal dataset raw files are unavailable under {root}")
    payload = []
    for path in sorted(files):
        payload.append(
            {
                "name": _relative(root, path),
                "size": path.stat().st_size,
                "sha256": file_sha256(path) if path in metadata_set else None,
            }
        )
    return (canonical_sha256(payload), len(files))


def _binding(
    dataset_id: str,
    version: str,
    root: Path,
    metadata_files: Iterable[Path] = (),
    data_roots: Iterable[Path] | None = None,
) -> dict[str, Any]:
    digest, count = _fingerprint(root, metadata_files, data_roots)
    return {
        "dataset_id": dataset_id,
        "version": version,
        "fingerprint": digest,
        "file_count": count,
        "root_kind": "runtime_external_data_root",
    }


def _binding_from_selected_paths(
    dataset_id: str,
    version: str,
    root: Path,
    paths: Sequence[Path],
    metadata_files: Iterable[Path] = (),
) -> dict[str, Any]:
    """Bind exactly the selected evaluation rows without scanning unrelated images."""
    metadata = tuple(path for path in metadata_files if path.is_file())
    selected = tuple(paths)
    if not selected:
        raise FileNotFoundError(f"{dataset_id} has no selected image paths")
    payload = [
        {"name": _relative(root, path), "size": path.stat().st_size, "sha256": None}
        for path in sorted(selected)
    ]
    payload.extend(
        {
            "name": _relative(root, path),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(metadata)
    )
    return {
        "dataset_id": dataset_id,
        "version": version,
        "fingerprint": canonical_sha256(payload),
        "file_count": len(payload),
        "root_kind": "runtime_external_data_root",
    }


def _view(
    *,
    dataset_id: str,
    version: str,
    class_names: tuple[str, ...],
    prompt_class_names: tuple[str, ...],
    records: Mapping[str, Sequence[SampleEntry]],
    paths: Mapping[str, Sequence[Path]],
    transform: Any,
    group_metadata: Mapping[str, str] | None = None,
    split_policy: Mapping[str, Any],
    raw_data_binding: Mapping[str, Any],
    evaluation_split: str,
) -> FormalDatasetView:
    if len(class_names) != len(prompt_class_names):
        raise ValueError("canonical and prompt class orders must have equal lengths")
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = tuple(records.get(split, ()))
        split_paths = tuple(paths.get(split, ()))
        if len(rows) != len(split_paths):
            raise ValueError(f"{dataset_id} {split} records and image paths differ")
        if rows:
            splits[split] = StableIdDataset(_RecordDataset(split_paths, rows, transform), rows)
        else:
            splits[split] = EmptyDataset()
    return FormalDatasetView(
        dataset_id=dataset_id,
        version=version,
        class_names=class_names,
        prompt_class_names=prompt_class_names,
        splits=splits,
        records={split: tuple(records.get(split, ())) for split in ("train", "val", "test")},
        group_metadata=dict(group_metadata or {}),
        split_policy=split_policy,
        raw_data_binding=raw_data_binding,
        evaluation_split=evaluation_split,
    )


def _resolve_root(
    dataset_id: str, project_root: Path, external_data_roots: Mapping[str, Path] | None
) -> Path:
    if external_data_roots and dataset_id in external_data_roots:
        root = Path(external_data_roots[dataset_id])
        if not root.exists():
            raise FileNotFoundError(f"external data root for {dataset_id} does not exist: {root}")
        return root
    candidates = (
        project_root / "data" / dataset_id,
        project_root / "data" / "formal-zero-shot-assets-v1" / dataset_id,
        project_root / dataset_id,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no runtime data root for {dataset_id}; supply external_data_roots[{dataset_id!r}]"
    )


def _row_value(row: Mapping[str, Any], *keys: str) -> str | None:
    normalized = {_normal(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(_normal(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _read_table(path: Path) -> tuple[dict[str, Any], ...]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"metadata row is not an object: {path}")
                rows.append(dict(value))
        return tuple(rows)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        delimiter = "\t" if path.suffix.lower() == ".tab" else ","
        return tuple(dict(row) for row in csv.DictReader(handle, delimiter=delimiter))


def _places365_class_names(categories_file: Path) -> tuple[str, ...]:
    """Replay the released repository's nested Places365 name parsing."""
    names: dict[int, str] = {}
    for line in categories_file.read_text(encoding="utf-8").splitlines():
        tokens = line.strip().split()
        if len(tokens) < 2 or not tokens[-1].isdigit():
            continue
        category_path = tokens[0]
        parts = category_path.split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid Places365 category path: {category_path!r}")
        names[int(tokens[-1])] = parts[2].replace("/", " ").replace("_", " ")
    if not names or tuple(sorted(names)) != tuple(range(len(names))):
        raise ValueError("Places365 categories must be a contiguous zero-based order")
    return tuple(names[index] for index in range(len(names)))


def _metadata_file(
    root: Path, *names: str, suffixes: tuple[str, ...] = (".csv", ".jsonl", ".tab")
) -> Path:
    wanted = {_normal(name) for name in names}
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and (_normal(path.stem) in wanted)
    )
    if not candidates:
        raise FileNotFoundError(f"metadata file not found below {root}: {sorted(wanted)}")
    return candidates[0]


class RESISC45Adapter:
    dataset_id = "resisc45"
    version = "nwpu-resisc45-project-official-splits-v1"
    evaluation_split = "test"
    prompt_class_names = _RESISC45_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def _split_file(self, split: str) -> Path:
        aliases = {
            "train": ("resisc45-train.txt", "train.txt", "train_list.txt", "trainval.txt"),
            "val": ("resisc45-val.txt", "val.txt", "valid.txt", "validation.txt"),
            "test": ("resisc45-test.txt", "test.txt", "test_list.txt"),
        }
        candidates = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.name.lower() in aliases[split]
        )
        if not candidates:
            raise FileNotFoundError(f"RESISC45 {split} split list is missing")
        return candidates[0]

    def _read_split(
        self, split: str, image_index: Mapping[str, Sequence[Path]]
    ) -> tuple[tuple[Path, SampleEntry], ...]:
        source = self._split_file(split)
        rows: list[tuple[Path, SampleEntry]] = []
        class_map = {_normal(name): index for index, name in enumerate(_RESISC45_CLASSES)}
        for line in source.read_text(encoding="utf-8").splitlines():
            tokens = line.strip().split()
            if not tokens or tokens[0].startswith("#"):
                continue
            path, relative = _find_image(self.root, tokens[0], image_index)
            class_name = path.parent.name
            key = _normal(class_name)
            if key not in class_map:
                raise ValueError(f"unknown RESISC45 class folder: {class_name}")
            entry = SampleEntry(
                sample_id=f"{split}/{relative}",
                label=class_map[key],
                split_source=f"project:resisc45_official:{source.name}",
            )
            rows.append((path, entry))
        if not rows:
            raise ValueError(f"RESISC45 {split} split list is empty: {source}")
        return tuple(rows)

    def load(self) -> FormalDatasetView:
        split_rows: dict[str, tuple[tuple[Path, SampleEntry], ...]] = {}
        image_index: dict[str, list[Path]] = {}
        for path in _images(self.root):
            image_index.setdefault(path.name, []).append(path)
        for split in ("train", "val", "test"):
            try:
                split_rows[split] = self._read_split(split, image_index)
            except FileNotFoundError:
                split_rows[split] = ()
        if not any(split_rows.values()):
            raise FileNotFoundError("RESISC45 has no readable official split list")
        records = {split: tuple(row[1] for row in rows) for split, rows in split_rows.items()}
        paths = {split: tuple(row[0] for row in rows) for split, rows in split_rows.items()}
        split_sources = tuple(
            self._split_file(split) for split in ("train", "val", "test") if split_rows[split]
        )
        binding = _binding(self.dataset_id, self.version, self.root, split_sources)
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=_RESISC45_CLASSES,
            prompt_class_names=_RESISC45_PROMPTS,
            records=records,
            paths=paths,
            transform=self.transform,
            split_policy={"id": "project_cached_official_split_lists", "sources": "relative"},
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class EuroSATAdapter:
    dataset_id = "eurosat"
    version = "eurosat-rgb-official-source-stratified-v1"
    evaluation_split = "test"
    prompt_class_names = _EUROSAT_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def load(self) -> FormalDatasetView:
        class_dirs = {
            _normal(path.name): path
            for path in self.root.rglob("*")
            if path.is_dir() and _normal(path.name) in {_normal(item) for item in _EUROSAT_CLASSES}
        }
        if len(class_dirs) != len(_EUROSAT_CLASSES):
            raise FileNotFoundError("EuroSAT RGB class folders are incomplete")
        records: dict[str, list[SampleEntry]] = {"train": [], "val": [], "test": []}
        paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        all_files: list[Path] = []
        all_labels: list[int] = []
        for label, class_name in enumerate(_EUROSAT_CLASSES):
            class_dir = class_dirs[_normal(class_name)]
            files = tuple(
                path
                for path in sorted(class_dir.rglob("*"))
                if path.suffix.lower() in _IMAGE_SUFFIXES
            )
            if not files:
                raise ValueError(f"EuroSAT class is empty: {class_name}")
            all_files.extend(files)
            all_labels.extend([label] * len(files))
        _train_indices, test_indices = _sklearn_stratified_split(
            all_labels, test_size=0.1, random_state=42
        )
        test_set = {int(index) for index in test_indices}
        for index, (path, label) in enumerate(zip(all_files, all_labels, strict=True)):
            class_name = _EUROSAT_CLASSES[label]
            split = "test" if index in test_set else "train"
            records[split].append(
                SampleEntry(
                    sample_id=f"{split}/{class_name}/{path.name}",
                    label=label,
                    split_source="official_source:eurosat_rgb:90_10_stratified_seed42",
                )
            )
            paths[split].append(path)
        binding = _binding(self.dataset_id, self.version, self.root)
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=_EUROSAT_CLASSES,
            prompt_class_names=_EUROSAT_PROMPTS,
            records=records,
            paths=paths,
            transform=self.transform,
            split_policy={"id": "official_source_stratified_90_10", "seed": 42},
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class Places365Adapter:
    dataset_id = "places365"
    version = "places365-validation-fifty-per-class-v1"
    evaluation_split = "val"

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def load(self) -> FormalDatasetView:
        category_candidates = sorted(self.root.rglob("categories_places365.txt"))
        val_candidates = sorted(self.root.rglob("places365_val.txt"))
        if not category_candidates or not val_candidates:
            raise FileNotFoundError(
                "Places365 categories_places365.txt or places365_val.txt is missing"
            )
        categories_file, val_file = (category_candidates[0], val_candidates[0])
        class_names = _places365_class_names(categories_file)
        categories = {index: name for index, name in enumerate(class_names)}
        image_index = {path.name: path for path in _images(self.root)}
        rows: list[SampleEntry] = []
        paths: list[Path] = []
        counts = {label: 0 for label in categories}
        for line in val_file.read_text(encoding="utf-8").splitlines():
            tokens = line.strip().split()
            if len(tokens) < 2 or not tokens[1].isdigit():
                continue
            filename, label = (tokens[0], int(tokens[1]))
            if label not in categories:
                raise ValueError(f"Places365 validation row has unknown label: {label}")
            if counts[label] >= 50:
                continue
            try:
                path = image_index[Path(filename).name]
            except KeyError as error:
                raise FileNotFoundError(
                    f"Places365 validation image is missing: {filename}"
                ) from error
            counts[label] += 1
            rows.append(
                SampleEntry(
                    sample_id=f"val/{path.name}",
                    label=label,
                    split_source="paper:places365_validation_all_50_per_class",
                )
            )
            paths.append(path)
        if not rows:
            raise ValueError("Places365 validation file contains no rows")
        binding = _binding(self.dataset_id, self.version, self.root, (categories_file, val_file))
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=class_names,
            prompt_class_names=class_names,
            records={"train": (), "val": tuple(rows), "test": ()},
            paths={"train": (), "val": tuple(paths), "test": ()},
            transform=self.transform,
            split_policy={
                "id": "validation_all_50_per_class",
                "source_split": "val",
                "class_counts": counts,
            },
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class HAM10000Adapter:
    dataset_id = "ham10000"
    version = "ham10000-huggingface-train-all-rows-v1"
    evaluation_split = "test"
    prompt_class_names = _HAM_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def load(self) -> FormalDatasetView:
        metadata = _metadata_file(self.root, "HAM10000_metadata", "metadata")
        rows = _read_table(metadata)
        code_to_label = {code: index for index, code in enumerate(_HAM_CODES)}
        image_index = {path.stem: path for path in _images(self.root)}
        entries: list[SampleEntry] = []
        paths: list[Path] = []
        for row in rows:
            code = (_row_value(row, "dx", "label", "class") or "").lower()
            image_id = _row_value(row, "image_id", "image", "id", "filename")
            if code not in code_to_label or image_id is None:
                continue
            stem = Path(image_id).stem
            try:
                path = image_index[stem]
            except KeyError as error:
                raise FileNotFoundError(f"HAM10000 image is missing: {image_id}") from error
            relative = _relative(self.root, path)
            entries.append(
                SampleEntry(
                    sample_id=f"test/{relative}",
                    label=code_to_label[code],
                    split_source="huggingface:HAM10000:train",
                )
            )
            paths.append(path)
        if not entries:
            raise ValueError("HAM10000 metadata contains no usable train rows")
        binding = _binding(self.dataset_id, self.version, self.root, (metadata,))
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=_HAM_CODES,
            prompt_class_names=self.prompt_class_names,
            records={"train": (), "val": (), "test": tuple(entries)},
            paths={"train": (), "val": (), "test": tuple(paths)},
            transform=self.transform,
            split_policy={"id": "huggingface_train_all_rows", "source": metadata.name},
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class PlantVillageAdapter:
    """PlantVillage color images with the released leaf-grouped split lists."""

    dataset_id = "plantvillage"
    version = "mohanty-plantvillage-color-leaf-grouped-v1"
    evaluation_split = "test"
    class_names = _PLANTVILLAGE_CLASSES
    prompt_class_names = _PLANTVILLAGE_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def _split_file(self, split: str) -> Path:
        preferred = self.root / "splits" / f"color_{split}.txt"
        if preferred.is_file():
            return preferred
        candidates = tuple(
            sorted(
                path
                for path in self.root.rglob(f"color_{split}.txt")
                if path.is_file() and "_hf" not in path.parts
            )
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"PlantVillage color_{split}.txt is missing or ambiguous below {self.root}"
            )
        return candidates[0]

    def _read_split(
        self, split: str, class_to_label: Mapping[str, int]
    ) -> tuple[tuple[SampleEntry, ...], tuple[Path, ...], Path]:
        split_file = self._split_file(split)
        entries: list[SampleEntry] = []
        paths: list[Path] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            path, relative = _find_image(self.root, value)
            class_name = path.parent.name
            if class_name not in class_to_label:
                raise ValueError(f"PlantVillage class is outside frozen roster: {class_name!r}")
            entries.append(
                SampleEntry(
                    sample_id=f"{split}/{relative}",
                    label=class_to_label[class_name],
                    split_source=f"mohanty:PlantVillage:color_{split}:leaf_grouped",
                )
            )
            paths.append(path)
        if not entries:
            raise ValueError(f"PlantVillage split is empty: {split_file}")
        return (tuple(entries), tuple(paths), split_file)

    def load(self) -> FormalDatasetView:
        class_to_label = {name: index for index, name in enumerate(self.class_names)}
        train_entries, train_paths, train_file = self._read_split("train", class_to_label)
        test_entries, test_paths, test_file = self._read_split("test", class_to_label)
        observed = {path.parent.name for path in train_paths + test_paths}
        if observed != set(self.class_names):
            raise ValueError(
                "PlantVillage split class roster differs from the frozen 38-class order"
            )
        binding = _binding_from_selected_paths(
            self.dataset_id,
            self.version,
            self.root,
            train_paths + test_paths,
            (train_file, test_file),
        )
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=self.class_names,
            prompt_class_names=self.prompt_class_names,
            records={"train": train_entries, "val": (), "test": test_entries},
            paths={"train": train_paths, "val": (), "test": test_paths},
            transform=self.transform,
            split_policy={
                "id": "mohanty_plantvillage_color_leaf_grouped_80_20",
                "source": "mohanty/PlantVillage",
                "configuration": "color",
            },
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class GTSRBAdapter:
    """Torchvision GTSRB adapter bound to the official train/test archives."""

    dataset_id = "gtsrb"
    version = "torchvision-gtsrb-official-train-test"
    evaluation_split = "test"
    class_names = tuple(f"class_{index:02d}" for index in range(43))
    prompt_class_names = _GTSRB_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def _load_split(
        self, split: str
    ) -> tuple[tuple[SampleEntry, ...], tuple[Path, ...], tuple[Path, ...]]:
        dataset = GTSRB(self.root, split=split, transform=self.transform, download=False)
        samples = tuple(((Path(path), int(label)) for path, label in dataset._samples))
        if not samples:
            raise ValueError(f"GTSRB {split} split is empty")
        base = Path(dataset._base_folder)
        entries = tuple(
            (
                SampleEntry(
                    sample_id=f"{split}/{_relative(base, path)}",
                    label=label,
                    split_source=f"torchvision:GTSRB:official:{split}",
                )
                for path, label in samples
            )
        )
        return (entries, tuple((path for path, _ in samples)), (base / "GT-final_test.csv",))

    def load(self) -> FormalDatasetView:
        train_entries, train_paths, train_metadata = self._load_split("train")
        test_entries, test_paths, test_metadata = self._load_split("test")
        labels = {row.label for row in train_entries + test_entries}
        if labels != set(range(43)):
            raise ValueError("GTSRB class IDs are not the official contiguous 0..42 roster")
        metadata = tuple(path for path in train_metadata + test_metadata if path.is_file())
        binding = _binding_from_selected_paths(
            self.dataset_id, self.version, self.root, train_paths + test_paths, metadata
        )
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=self.class_names,
            prompt_class_names=self.prompt_class_names,
            records={"train": train_entries, "val": (), "test": test_entries},
            paths={"train": train_paths, "val": (), "test": test_paths},
            transform=self.transform,
            split_policy={"id": "torchvision_gtsrb_official_train_test"},
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class Camelyon17Adapter:
    dataset_id = "camelyon17"
    version = "camelyon17-wilds-official-ood-val-center1-v1"
    evaluation_split = "val"
    prompt_class_names = _CAMELYON_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def load(self) -> FormalDatasetView:
        metadata = _metadata_file(self.root, "metadata", "camelyon17_metadata", "camelyon17")
        rows = _read_table(metadata)
        entries: list[SampleEntry] = []
        paths: list[Path] = []
        groups: dict[str, str] = {}
        for row in rows:
            center = (_row_value(row, "center", "hospital", "hospital_id") or "").lower()
            center_number = re.search("\\d+", center)
            if center_number is None or int(center_number.group(0)) != 1:
                continue
            image_value = _row_value(
                row, "image", "image_path", "path", "relative_path", "filename"
            )
            if image_value is None:
                image_value = _row_value(row, "patch_id", "sample_id", "image_id")
            if image_value is not None:
                path, relative = _find_image(self.root, image_value)
            else:
                patient = _row_value(row, "patient")
                node = _row_value(row, "node")
                x_coord = _row_value(row, "x_coord", "x")
                y_coord = _row_value(row, "y_coord", "y")
                if None in {patient, node, x_coord, y_coord}:
                    continue
                patient_token = str(patient)
                node_token = str(node)
                filename = (
                    f"patch_patient_{patient_token}_node_{node_token}_x_{x_coord}_y_{y_coord}.png"
                )
                relative_dir = Path("patches") / f"patient_{patient_token}_node_{node_token}"
                candidate = self.root / relative_dir / filename
                if not candidate.is_file():
                    matches = tuple(self.root.rglob(filename))
                    if len(matches) != 1:
                        raise FileNotFoundError(f"CAMELYON17 patch is missing: {filename}")
                    candidate = matches[0]
                path, relative = (candidate, _relative(self.root, candidate))
            raw_label = _row_value(row, "tumor", "label", "target", "class")
            try:
                label = int(raw_label or "0")
            except ValueError:
                label = 1 if _normal(raw_label) in {"tumor", "metastatic", "metastatictumor"} else 0
            label = 1 if label else 0
            sample_id = f"val/{relative}"
            entries.append(
                SampleEntry(sample_id=sample_id, label=label, split_source="wilds:ood_val:center1")
            )
            paths.append(path)
            patient = _row_value(row, "patient") or ""
            slide = _row_value(row, "slide") or ""
            groups[sample_id] = json.dumps(
                {"patient": patient, "slide": slide, "center": center},
                sort_keys=True,
                separators=(",", ":"),
            )
        if not entries:
            raise ValueError("CAMELYON17 metadata contains no official OOD-val center1 rows")
        binding = _binding_from_selected_paths(
            self.dataset_id, self.version, self.root, paths, (metadata,)
        )
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=("normal", "tumor"),
            prompt_class_names=self.prompt_class_names,
            records={"train": (), "val": tuple(entries), "test": ()},
            paths={"train": (), "val": tuple(paths), "test": ()},
            transform=self.transform,
            group_metadata=groups,
            split_policy={"id": "wilds_official_ood_val_center1", "center": 1},
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


class OCT2017Adapter:
    dataset_id = "oct2017"
    version = "kermany-oct2017-frozen-group-disjoint-val-v1"
    evaluation_split = "val"
    prompt_class_names = _OCT_PROMPTS

    def __init__(self, root: str | Path, *, transform: Any = None) -> None:
        self.root = Path(root)
        self.transform = transform

    def _assignment_file(self) -> Path:
        exact = tuple(self.root.rglob("randomized_patient_id_group_assignment.jsonl"))
        if exact:
            return sorted(exact)[0]
        candidates = sorted(
            path
            for path in self.root.rglob("*.jsonl")
            if "assign" in path.stem.lower() or "split" in path.stem.lower()
        )
        for path in candidates:
            if "test" not in path.stem.lower() and "lock" not in path.stem.lower():
                return path
        raise FileNotFoundError("OCT2017 frozen validation assignment JSONL is missing")

    def load(self) -> FormalDatasetView:
        assignment = self._assignment_file()
        rows = _read_table(assignment)
        class_map = {_normal(name): index for index, name in enumerate(_OCT_CLASSES)}
        validation_groups = {
            str(_row_value(row, "randomized_patient_id_group"))
            for row in rows
            if (_row_value(row, "recommended_split", "split") or "").lower() == "val"
        }
        train_root = self.root / "OCT2017" / "train"
        if not train_root.is_dir():
            raise FileNotFoundError("OCT2017 staged train tree is missing")
        entries: list[SampleEntry] = []
        paths: list[Path] = []
        groups: dict[str, str] = {}
        for class_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
            key = _normal(class_dir.name)
            if key not in class_map:
                continue
            for path in _images(class_dir):
                parts = path.stem.split("-")
                if len(parts) < 2 or parts[1] not in validation_groups:
                    continue
                relative = _relative(self.root, path)
                sample_id = f"val/{relative}"
                entries.append(
                    SampleEntry(
                        sample_id=sample_id,
                        label=class_map[key],
                        split_source="project:oct2017:frozen_val_assignment",
                    )
                )
                paths.append(path)
                groups[sample_id] = parts[1]
        if not entries:
            raise ValueError("OCT2017 assignment contains no validation rows")
        binding = _binding_from_selected_paths(
            self.dataset_id, self.version, self.root, paths, (assignment,)
        )
        return _view(
            dataset_id=self.dataset_id,
            version=self.version,
            class_names=_OCT_CLASSES,
            prompt_class_names=self.prompt_class_names,
            records={"train": (), "val": tuple(entries), "test": ()},
            paths={"train": (), "val": tuple(paths), "test": ()},
            transform=self.transform,
            group_metadata=groups,
            split_policy={
                "id": "frozen_group_disjoint_val",
                "assignment": assignment.name,
                "locked_test_excluded": True,
            },
            raw_data_binding=binding,
            evaluation_split=self.evaluation_split,
        )


FORMAL_ADAPTERS: dict[str, type[Any]] = {
    "resisc45": RESISC45Adapter,
    "eurosat": EuroSATAdapter,
    "places365": Places365Adapter,
    "ham10000": HAM10000Adapter,
    "camelyon17": Camelyon17Adapter,
    "oct2017": OCT2017Adapter,
    "plantvillage": PlantVillageAdapter,
    "gtsrb": GTSRBAdapter,
}


def load_formal_dataset(
    dataset_id: str,
    *,
    project_root: Path,
    external_data_roots: Mapping[str, Path] | None = None,
    transform: Any = None,
) -> FormalDatasetView:
    """Load one formal dataset without persisting machine-specific paths."""
    if dataset_id in FORMAL_ADAPTERS:
        root = _resolve_root(dataset_id, Path(project_root), external_data_roots)
        return FORMAL_ADAPTERS[dataset_id](root, transform=transform).load()
    if dataset_id in {"dtd", "oxford_pets"}:
        root = _resolve_root(dataset_id, Path(project_root), external_data_roots)
        view = create_dataset_adapter(dataset_id, root, transform=transform, download=False).load()
        prompt_names = (
            OFFICIAL_DTD_CLASS_NAMES if dataset_id == "dtd" else OFFICIAL_OXFORD_PETS_CLASS_NAMES
        )
        if len(view.class_names) != len(prompt_names):
            raise ValueError(f"{dataset_id} canonical/prompt class counts differ")
        return FormalDatasetView(
            dataset_id=view.dataset_id,
            version=view.version,
            class_names=view.class_names,
            prompt_class_names=prompt_names,
            splits=view.splits,
            records=view.records,
            group_metadata={},
            split_policy={"id": "existing_clean_room_adapter", "source": view.version},
            raw_data_binding={
                "dataset_id": view.dataset_id,
                "version": view.version,
                "fingerprint": view.raw_data_fingerprint,
                "file_count": None,
                "root_kind": "runtime_external_data_root",
            },
            evaluation_split="test",
        )
    raise NotImplementedError(
        f"formal zero-shot adapter is not implemented for "
        f"{dataset_id!r}; use the existing preload path"
    )


def formal_prompt_class_names(
    dataset_id: str, *, project_root: Path, external_data_roots: Mapping[str, Path] | None = None
) -> tuple[str, ...]:
    """Return prompt names without scanning or hashing the image inventory."""
    fixed: dict[str, tuple[str, ...]] = {
        "resisc45": _RESISC45_PROMPTS,
        "eurosat": _EUROSAT_PROMPTS,
        "ham10000": _HAM_PROMPTS,
        "camelyon17": _CAMELYON_PROMPTS,
        "oct2017": _OCT_PROMPTS,
        "dtd": OFFICIAL_DTD_CLASS_NAMES,
        "oxford_pets": OFFICIAL_OXFORD_PETS_CLASS_NAMES,
        "plantvillage": _PLANTVILLAGE_PROMPTS,
        "gtsrb": _GTSRB_PROMPTS,
    }
    if dataset_id in fixed:
        return fixed[dataset_id]
    root = _resolve_root(dataset_id, Path(project_root), external_data_roots)
    if dataset_id == "places365":
        candidates = sorted(root.rglob("categories_places365.txt"))
        if not candidates:
            raise FileNotFoundError("Places365 categories_places365.txt is missing")
        return _places365_class_names(candidates[0])
    raise NotImplementedError(f"prompt class names are not registered for {dataset_id!r}")


def formal_canonical_class_names(
    dataset_id: str, *, project_root: Path, external_data_roots: Mapping[str, Path] | None = None
) -> tuple[str, ...]:
    """Return the label-index class order without enumerating image files."""
    fixed: dict[str, tuple[str, ...]] = {
        "resisc45": _RESISC45_CLASSES,
        "eurosat": _EUROSAT_CLASSES,
        "ham10000": _HAM_CODES,
        "camelyon17": ("normal", "tumor"),
        "oct2017": _OCT_CLASSES,
        "dtd": tuple(name.removesuffix(" texture") for name in OFFICIAL_DTD_CLASS_NAMES),
        "oxford_pets": tuple(
            name.removesuffix(", a type of pet") for name in OFFICIAL_OXFORD_PETS_CLASS_NAMES
        ),
        "plantvillage": _PLANTVILLAGE_CLASSES,
        "gtsrb": tuple(f"class_{index:02d}" for index in range(43)),
    }
    if dataset_id in fixed:
        return fixed[dataset_id]
    if dataset_id == "places365":
        return formal_prompt_class_names(
            dataset_id, project_root=project_root, external_data_roots=external_data_roots
        )
    raise NotImplementedError(f"canonical class names are not registered for {dataset_id!r}")


CAMELYON17Adapter = Camelyon17Adapter
