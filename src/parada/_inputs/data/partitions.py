from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _client_names(num_clients: int) -> tuple[str, ...]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    width = max(4, len(str(num_clients - 1)))
    return tuple(f"client_{index:0{width}d}" for index in range(num_clients))


def split_iid(sample_ids: Sequence[str], *, num_clients: int, seed: int) -> dict[str, list[str]]:
    names = _client_names(num_clients)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(sample_ids))
    chunks = np.array_split(permutation, num_clients)
    partition = {
        name: [str(sample_ids[int(index)]) for index in chunk.tolist()]
        for name, chunk in zip(names, chunks, strict=True)
    }
    validate_partition_payload(partition, sample_ids)
    return partition


def split_dirichlet(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    *,
    num_clients: int,
    alpha: float,
    seed: int,
    minimum_size: int = 1,
    max_attempts: int = 1000,
) -> dict[str, list[str]]:
    names = _client_names(num_clients)
    if len(sample_ids) != len(labels):
        raise ValueError("sample_ids and labels must have equal lengths")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if minimum_size < 0 or len(sample_ids) < num_clients * minimum_size:
        raise ValueError("minimum_size cannot be satisfied")
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.size and int(label_array.min()) < 0:
        raise ValueError("labels must be nonnegative")
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(label_array)
    for _ in range(max_attempts):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in unique_labels.tolist():
            indices = np.flatnonzero(label_array == class_id)
            rng.shuffle(indices)
            probabilities = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            allocation = rng.multinomial(len(indices), probabilities)
            cursor = 0
            for client_index, count in enumerate(allocation.tolist()):
                buckets[client_index].extend(indices[cursor : cursor + count].tolist())
                cursor += count
        if min((len(bucket) for bucket in buckets), default=0) < minimum_size:
            continue
        for bucket in buckets:
            rng.shuffle(bucket)
        partition = {
            name: [str(sample_ids[index]) for index in bucket]
            for name, bucket in zip(names, buckets, strict=True)
        }
        validate_partition_payload(partition, sample_ids)
        return partition
    raise RuntimeError(f"failed to satisfy minimum_size after {max_attempts} attempts")


def validate_partition_payload(
    partition: Mapping[str, Sequence[str]], expected_sample_ids: Sequence[str]
) -> None:
    expected = [str(item) for item in expected_sample_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_sample_ids must be unique")
    flattened = [str(item) for client in sorted(partition) for item in partition[client]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("partition contains duplicate sample IDs")
    missing = sorted(set(expected) - set(flattened))
    unexpected = sorted(set(flattened) - set(expected))
    if missing or unexpected:
        raise ValueError(f"partition coverage mismatch: missing={missing}, unexpected={unexpected}")
