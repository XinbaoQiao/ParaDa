from __future__ import annotations

import pytest
from parada._inputs.data.partitions import split_dirichlet, split_iid, validate_partition_payload


def test_iid_partition_is_deterministic_complete_and_unique() -> None:
    sample_ids = [f"train/{index:03d}" for index in range(23)]
    first = split_iid(sample_ids, num_clients=5, seed=9)
    second = split_iid(sample_ids, num_clients=5, seed=9)
    assert first == second
    validate_partition_payload(first, sample_ids)
    sizes = [len(first[key]) for key in sorted(first)]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_partition_is_deterministic_and_satisfies_minimum() -> None:
    sample_ids = [f"train/{index:03d}" for index in range(120)]
    labels = [index % 6 for index in range(120)]
    first = split_dirichlet(sample_ids, labels, num_clients=8, alpha=0.3, seed=11, minimum_size=3)
    second = split_dirichlet(sample_ids, labels, num_clients=8, alpha=0.3, seed=11, minimum_size=3)
    assert first == second
    assert min(map(len, first.values())) >= 3
    validate_partition_payload(first, sample_ids)


def test_partition_validator_rejects_duplicate_or_missing_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_partition_payload({"a": ["x"], "b": ["x"]}, ["x", "y"])
    with pytest.raises(ValueError, match="coverage"):
        validate_partition_payload({"a": ["x"]}, ["x", "y"])


def test_impossible_minimum_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be satisfied"):
        split_dirichlet(["a"], [0], num_clients=2, alpha=1.0, seed=0, minimum_size=1)
