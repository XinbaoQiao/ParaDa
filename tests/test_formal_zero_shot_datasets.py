from __future__ import annotations

import json
from pathlib import Path

from parada._inputs.data.formal_zero_shot_datasets import (
    FORMAL_ZERO_SHOT_DATASET_IDS,
    CAMELYON17Adapter,
    HAM10000Adapter,
    OCT2017Adapter,
    Places365Adapter,
    PlantVillageAdapter,
    RESISC45Adapter,
    formal_canonical_class_names,
    formal_prompt_class_names,
    load_formal_dataset,
)
from parada._inputs.data.registry import DATASET_REGISTRY
from PIL import Image


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path)


def test_registry_exposes_formal_blocks_and_medical_protocols() -> None:
    assert len(FORMAL_ZERO_SHOT_DATASET_IDS) == 19
    assert FORMAL_ZERO_SHOT_DATASET_IDS[:5] == (
        "cifar10",
        "cifar100",
        "stanforddogs",
        "oxford_pets",
        "cub2011",
    )
    assert DATASET_REGISTRY["eurosat"].split_policy == "official_source_stratified_90_10_seed42"
    assert DATASET_REGISTRY["camelyon17"].primary_metric == "balanced_accuracy"
    assert DATASET_REGISTRY["oct2017"].split_policy == "project:frozen_group_disjoint_val_seed3407"
    assert DATASET_REGISTRY["plantvillage"].class_count == 38
    assert DATASET_REGISTRY["gtsrb"].class_count == 43


def test_resisc45_reads_cached_split_lists_with_portable_ids(tmp_path: Path) -> None:
    root = tmp_path / "resisc45"
    _image(root / "airplane" / "a.jpg")
    _image(root / "airport" / "b.jpg")
    (root / "splits").mkdir()
    (root / "splits" / "resisc45-train.txt").write_text("airplane/a.jpg\n", encoding="utf-8")
    (root / "splits" / "resisc45-test.txt").write_text("airport/b.jpg\n", encoding="utf-8")

    view = RESISC45Adapter(root).load()

    assert view.records["train"][0].sample_id == "train/airplane/a.jpg"
    assert view.records["test"][0].sample_id == "test/airport/b.jpg"
    assert view.prompt_class_names[0] == "airplane (aerial photo)"
    assert view.split_policy["id"] == "project_cached_official_split_lists"
    assert "/" not in view.raw_data_binding["fingerprint"]


def test_eurosat_uses_seeded_stratified_ninety_ten_split(tmp_path: Path) -> None:
    root = tmp_path / "eurosat"
    for class_name in (
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
    ):
        for index in range(10):
            _image(root / class_name / f"{index}.jpg")

    view = load_formal_dataset(
        "eurosat", project_root=tmp_path, external_data_roots={"eurosat": root}
    )

    assert len(view.records["train"]) == 90
    assert len(view.records["test"]) == 10
    assert view.split_policy == {"id": "official_source_stratified_90_10", "seed": 42}
    assert all(row.split_source.endswith("seed42") for row in view.records["test"])
    assert view.prompt_class_names[0] == "annual crop land  (aerial photo)"


def test_places365_caps_validation_at_fifty_images_per_class(tmp_path: Path) -> None:
    root = tmp_path / "places"
    rows: list[str] = []
    for label in range(2):
        for index in range(51):
            name = f"Places365_val_{label:03d}_{index:03d}.jpg"
            _image(root / name)
            rows.append(f"{name} {label}\n")
    (root / "categories_places365.txt").write_text("/a/airport 0\n/a/forest 1\n", encoding="utf-8")
    (root / "places365_val.txt").write_text("".join(rows), encoding="utf-8")

    view = Places365Adapter(root).load()

    assert view.evaluation_split == "val"
    assert len(view.records["val"]) == 100
    assert {row.label for row in view.records["val"]} == {0, 1}
    assert view.split_policy["class_counts"] == {0: 50, 1: 50}
    assert view.prompt_class_names == ("airport", "forest")


def test_prompt_names_read_places_categories_without_image_scan(tmp_path: Path) -> None:
    root = tmp_path / "places"
    root.mkdir()
    (root / "categories_places365.txt").write_text(
        "/a/airport_terminal 0\n/a/arena/hockey 1\n", encoding="utf-8"
    )

    assert formal_prompt_class_names(
        "places365", project_root=tmp_path, external_data_roots={"places365": root}
    ) == ("airport terminal", "arena hockey")


def test_ham10000_freezes_hf_train_rows_and_expanded_prompt_names(tmp_path: Path) -> None:
    root = tmp_path / "ham"
    _image(root / "images" / "n1.jpg")
    _image(root / "images" / "m1.jpg")
    (root / "HAM10000_metadata.tab").write_text("image_id\tdx\nn1\tnv\nm1\tmel\n", encoding="utf-8")

    view = HAM10000Adapter(root).load()

    assert view.class_names == ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
    assert view.prompt_class_names[1] == "Melanoma (malignant skin cancer)"
    assert {row.split_source for row in view.records["test"]} == {"huggingface:HAM10000:train"}


def test_plantvillage_reads_released_color_leaf_grouped_splits(tmp_path: Path) -> None:
    root = tmp_path / "plantvillage"
    class_names = PlantVillageAdapter.class_names
    train_lines: list[str] = []
    test_lines: list[str] = []
    for index, class_name in enumerate(class_names):
        train_name = f"raw/color/{class_name}/train_{index}.jpg"
        test_name = f"raw/color/{class_name}/test_{index}.jpg"
        _image(root / train_name)
        _image(root / test_name)
        train_lines.append(train_name + "\n")
        test_lines.append(test_name + "\n")
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "color_train.txt").write_text("".join(train_lines), encoding="utf-8")
    (root / "splits" / "color_test.txt").write_text("".join(test_lines), encoding="utf-8")

    view = PlantVillageAdapter(root).load()

    assert len(view.records["train"]) == 38
    assert len(view.records["test"]) == 38
    assert view.class_names[0] == "Apple___Apple_scab"
    assert view.split_policy["id"] == "mohanty_plantvillage_color_leaf_grouped_80_20"


def test_camelyon17_selects_only_ood_val_center1_and_keeps_group_metadata(tmp_path: Path) -> None:
    root = tmp_path / "camelyon"
    _image(root / "patches" / "patient_004_node_1" / "patch_patient_004_node_1_x_10_y_20.png")
    _image(root / "patches" / "patient_005_node_1" / "patch_patient_005_node_1_x_10_y_20.png")
    (root / "metadata.csv").write_text(
        ",patient,node,x_coord,y_coord,tumor,slide,center,split\n"
        "0,004,1,10,20,1,slide-a,1,0\n"
        "1,005,1,10,20,0,slide-b,2,0\n",
        encoding="utf-8",
    )

    view = CAMELYON17Adapter(root).load()

    assert len(view.records["val"]) == 1
    sample_id = view.records["val"][0].sample_id
    assert '"patient":"004"' in view.group_metadata[sample_id]
    assert view.prompt_class_names == ("normal lymph-node tissue", "metastatic tumor tissue")


def test_oct2017_uses_frozen_val_assignment_and_excludes_locked_test(tmp_path: Path) -> None:
    root = tmp_path / "oct"
    _image(root / "OCT2017" / "train" / "CNV" / "CNV-group-1.jpeg")
    _image(root / "OCT2017" / "train" / "DME" / "DME-group2-1.jpeg")
    assignment = root / "oct2017_val_assignments.jsonl"
    assignment.write_text(
        json.dumps(
            {
                "randomized_patient_id_group": "group",
                "recommended_split": "val",
                "image_count_vector": {"CNV": 1, "DME": 0, "DRUSEN": 0, "NORMAL": 0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "randomized_patient_id_group": "group2",
                "recommended_split": "test",
                "image_count_vector": {"CNV": 0, "DME": 1, "DRUSEN": 0, "NORMAL": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    view = OCT2017Adapter(root).load()

    assert len(view.records["val"]) == 1
    assert view.records["val"][0].label == 0
    assert view.group_metadata[view.records["val"][0].sample_id] == "group"
    assert view.split_policy["locked_test_excluded"] is True


def test_lightweight_prompt_and_canonical_names_match_released_source(
    tmp_path: Path,
) -> None:
    assert formal_prompt_class_names("dtd", project_root=tmp_path)[0] == "banded texture"
    assert formal_canonical_class_names("dtd", project_root=tmp_path)[0] == "banded"
    assert formal_prompt_class_names("oxford_pets", project_root=tmp_path)[0] == (
        "Abyssinian, a type of pet"
    )
    assert formal_canonical_class_names("oxford_pets", project_root=tmp_path)[0] == "Abyssinian"
    assert formal_prompt_class_names("ham10000", project_root=tmp_path)[0] == (
        "Melanocytic nevi (benign moles)"
    )
