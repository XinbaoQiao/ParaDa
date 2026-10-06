from __future__ import annotations

from collections.abc import Callable

from parada._inputs.data.bmpl_reference_manifests import (
    CanonicalTrainManifest,
    build_artbench_train_manifest,
    build_cifar10_train_manifest,
    build_cifar100_train_manifest,
    build_cub2011_train_manifest,
    build_flowers102_train_manifest,
    build_food101_train_manifest,
    build_imagenet_1k_train_manifest,
    build_imagenet_100_train_manifest,
    build_spawrious_train_manifest,
    build_stanforddogs_train_manifest,
    build_waterbirds_train_manifest,
)
from parada._inputs.data.bmpl_reference_records import OrderedRecordSet

REFERENCE_DATA_SUBDIRS = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "spawrious": "spawrious",
    "stanforddogs": "StanfordDogs",
    "food101": "food101",
    "artbench": "artbench",
    "imagenet-1k": "imagenet",
    "imagenet-100": "imagenet",
    "waterbirds": "waterbirds",
    "cub2011": "cub",
    "flowers102": "flowers102",
}
PathBuilder = Callable[..., CanonicalTrainManifest]
PATH_BUILDERS: dict[str, PathBuilder] = {
    "spawrious": build_spawrious_train_manifest,
    "stanforddogs": build_stanforddogs_train_manifest,
    "food101": build_food101_train_manifest,
    "artbench": build_artbench_train_manifest,
    "imagenet-1k": build_imagenet_1k_train_manifest,
    "imagenet-100": build_imagenet_100_train_manifest,
    "waterbirds": build_waterbirds_train_manifest,
    "cub2011": build_cub2011_train_manifest,
    "flowers102": build_flowers102_train_manifest,
}


def _build_manifest(records: OrderedRecordSet) -> CanonicalTrainManifest:
    identity = dict(records.dataset_asset_identity)
    if records.dataset_id == "cifar10":
        return build_cifar10_train_manifest(
            [row.label for row in records.records], dataset_asset_identity=identity
        )
    if records.dataset_id == "cifar100":
        return build_cifar100_train_manifest(
            [row.label for row in records.records], dataset_asset_identity=identity
        )
    builder = PATH_BUILDERS[records.dataset_id]
    return builder(records.as_manifest_records(), dataset_asset_identity=identity)
