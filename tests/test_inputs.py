import json
from pathlib import Path

import numpy as np
import pytest
import torch
from parada import prepare, source
from parada._inputs.data.partitions import split_dirichlet


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_fresh_source_seed_covers_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "MLP_EPOCHS", 2)
    g = torch.Generator().manual_seed(81)
    text, visual = torch.randn(20, 5, generator=g), torch.randn(20, 4, generator=g)
    states = []
    for i in range(2):
        torch.manual_seed(100 + i)
        _, receipt = source.train_or_load_source_model(
            text, visual, seed=42, device=torch.device("cpu"), output_root=tmp_path / str(i)
        )
        states.append(receipt["state_sha256"])
    assert states[0] == states[1]


def test_catalog_matches_installed_model_configs():
    import timm

    for row in prepare.CATALOG["models"].values():
        cfg = timm.get_pretrained_cfg(row["model"]).to_dict()
        for key, expected in row["preprocessing"].items():
            actual = list(cfg[key]) if isinstance(cfg[key], tuple) else cfg[key]
            assert actual == expected
        assert len(row["sha256"]) == 64 and len(row["revision"]) == 40


def test_model_hash_rejected_before_loader(tmp_path, monkeypatch):
    import parada._inputs.models.vision as vision

    monkeypatch.setattr(vision, "TimmVisionEncoder", lambda *a, **k: pytest.fail("loader invoked"))
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="mismatched"):
        prepare.vision("vit_tiny", checkpoint, "cpu")


def test_text_preserves_raw_target_and_native_source_normalization():
    class Clip:
        @staticmethod
        def tokenize(texts, truncate):
            assert truncate is mode[0]
            return torch.arange(len(texts)).reshape(-1, 1)

    class Model:
        def encode_text(self, tokens):
            return torch.tensor([[1.1, 2.3]], dtype=torch.float16).repeat(len(tokens), 1)

    mode = [False]
    raw = prepare.encode_text(Clip, Model(), ["a", "b", "c"], "cpu", batch_size=2)
    torch.testing.assert_close(
        raw, torch.tensor([[1.1, 2.3]], dtype=torch.float16).float().repeat(3, 1)
    )
    mode[0] = True
    normalized = prepare.encode_text(Clip, Model(), ["a"], "cpu", normalize=True)
    v = torch.tensor([[1.1, 2.3]], dtype=torch.float16)
    torch.testing.assert_close(
        normalized, (v / v.norm(dim=-1, keepdim=True)).float(), rtol=0, atol=0
    )


def test_description_request_response_flow_and_order_rejection(tmp_path):
    from parada.descriptions import DescriptionValidationError

    index = tmp_path / "index.json"
    index.write_text(json.dumps({"class_names": ["red fox", "sea turtle"]}))
    requests = tmp_path / "requests"
    prepare.main(["description-prompts", "--index", str(index), "--output", str(requests)])
    request = json.loads((requests / "batch-000.json").read_text())
    assert request["model"] == "gpt-4o-2024-11-20"
    assert "1. red fox\n2. sea turtle" in request["messages"][1]["content"]
    assert "seed" not in request and "temperature" not in request
    rows = [
        {
            "class_name": name,
            "appearance": "A small animal has distinctive shapes and clearly visible features",
            "functionality": "It moves through its habitat while seeking food and shelter",
            "environment": (
                "Its natural environment supports daily movement and varied feeding behaviors"
            ),
        }
        for name in ("red fox", "sea turtle")
    ]
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"descriptions": rows}),
                        }
                    }
                ]
            }
        )
    )
    output = tmp_path / "descriptions.json"
    argv = ["collect-descriptions", "--index", str(index), "--responses", str(response)]
    prepare.main([*argv, "--output", str(output)])
    assert json.loads(output.read_text())["descriptions"] == rows
    response.write_text(json.dumps({"descriptions": rows[::-1]}))
    with pytest.raises(DescriptionValidationError, match="class order"):
        prepare.main([*argv, "--output", str(tmp_path / "wrong.json")])


def test_sampling_nested_support_shared_queries_and_global_owners():
    train = torch.arange(8).repeat_interleave(20)
    query = torch.arange(8).repeat_interleave(4)
    for e in range(200):
        one = prepare.select_episode(train, query, 8, 1, 42, "5way", e)
        five = prepare.select_episode(train, query, 8, 5, 42, "5way", e)
        assert one[0] == five[0] and torch.equal(one[2], five[2])
        assert set(one[1].tolist()) <= set(five[1].tolist())
        assert len(one[2]) == 5 and len(one[1]) == 5
        assert set(train[one[1]].tolist()) == set(one[0])
    ids = [f"train/{i}" for i in range(len(train))]
    a = split_dirichlet(ids, train.tolist(), num_clients=10, alpha=0.1, seed=3407)
    b = split_dirichlet(ids, train.tolist(), num_clients=10, alpha=0.1, seed=3407)
    assert a == b and sorted(x for rows in a.values() for x in rows) == sorted(ids)


def test_real_image_adapter_split_and_missing_metadata(tmp_path):
    from PIL import Image

    root = tmp_path / "eurosat"
    classes = (
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
    for name in classes:
        (root / name).mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (4, 4), (i, 20, 30)).save(root / name / f"{i}.jpg")
    _, a = prepare.dataset_index("eurosat", tmp_path)
    _, b = prepare.dataset_index("eurosat", tmp_path)
    assert a == b
    assert len(a["splits"]["train"]) == 90 and len(a["splits"]["test"]) == 10
    assert not ({r["id"] for r in a["splits"]["train"]} & {r["id"] for r in a["splits"]["test"]})
    (tmp_path / "resisc45").mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        prepare.dataset_index("resisc45", tmp_path)


def test_portable_cache_episode_adaptation_and_evaluation(tmp_path, monkeypatch):
    from parada.cli import main
    from PIL import Image
    from safetensors.torch import load_file

    root = tmp_path / "eurosat"
    names = (
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
    for c, name in enumerate(names):
        (root / name).mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (4, 4), (20 + i, 10 + c, 30)).save(root / name / f"{i}.jpg")

    class Encoder:
        @staticmethod
        def transform(image):
            return torch.tensor(np.array(image)[0, 0].copy(), dtype=torch.float32)

        @staticmethod
        def encode(images):
            return torch.cat([images, torch.ones(len(images), 1)], dim=1)

    monkeypatch.setattr(prepare, "vision", lambda *a: Encoder())
    cache = tmp_path / "cache"
    prepare.prepare_dataset("eurosat", tmp_path, "vit_tiny", Path("unused"), cache, "cpu", 16)
    info, _ = prepare.read_cache(cache)
    g = torch.Generator().manual_seed(10)
    source_path, target = tmp_path / "source.safetensors", tmp_path / "target.safetensors"
    source.write_once_safetensors(
        source_path,
        {
            "source_text": torch.randn(20, 5, generator=g),
            "source_visual": torch.randn(20, 4, generator=g),
        },
        {},
    )
    source.write_once_safetensors(target, {"target_views": torch.randn(10, 3, 5, generator=g)}, {})
    si = {
        "tensor_file_sha256": source.file_sha256(source_path),
        "vision_sha256": info["vision_sha256"],
        "text_sha256": "fixture",
        "preprocessing": info["preprocessing"],
    }
    ti = {
        "tensor_file_sha256": source.file_sha256(target),
        "text_sha256": "fixture",
        "class_names": info["class_names"],
    }
    source.write_once_json(Path(str(source_path) + ".json"), si)
    source.write_once_json(Path(str(target) + ".json"), ti)
    ep = tmp_path / "episode"
    prepare.prepare_episode(cache, source_path, target, ep, 1, 42, "fullway", 0)
    assert (
        sum(
            len(load_file(str(ep / f"client-{i}.safetensors"))["support_labels"]) for i in range(10)
        )
        == 10
    )
    monkeypatch.setattr(source, "MLP_EPOCHS", 2)
    checkpoint = tmp_path / "model"
    main(["train", "--source", str(source_path), "--checkpoint", str(checkpoint)])
    classifier = ep / "classifier.safetensors"
    main(
        [
            "adapt",
            "--source",
            str(source_path),
            "--checkpoint",
            str(checkpoint),
            "--target",
            str(ep / "target.safetensors"),
            "--clients",
            str(ep),
            "--k",
            "1",
            "--output",
            str(classifier),
        ]
    )
    predictions = ep / "predictions.safetensors"
    from safetensors import safe_open

    with safe_open(str(classifier), framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
    assert metadata["method"] == "parada-prototype" and "rounds" not in metadata
    audit = json.loads(metadata["prototype_audit"])
    assert audit["communication_rounds"] == 1 and audit["server_fit"]["steps"] == 100
    package = Path(prepare.__file__).parent
    assert metadata["method_config_sha256"] == source.file_sha256(package / "method_config.json")
    assert metadata["method_source_sha256"] == source.file_sha256(package / "federated.py")
    main(
        [
            "predict",
            "--features",
            str(ep / "query.safetensors"),
            "--classifier",
            str(classifier),
            "--output",
            str(predictions),
        ]
    )
    prepare.main(
        [
            "evaluate",
            "--predictions",
            str(predictions),
            "--labels",
            str(ep / "labels.safetensors"),
            "--output",
            str(ep / "metrics.json"),
        ]
    )
    assert json.loads((ep / "metrics.json").read_text())["query_count"] == 10
    ti["class_names"] = ti["class_names"][::-1]
    Path(str(target) + ".json").write_text(json.dumps(ti))
    with pytest.raises(ValueError, match="class order"):
        prepare.prepare_episode(cache, source_path, target, tmp_path / "bad", 1, 42, "fullway", 0)
    info["class_names"] = info["class_names"][::-1]
    (cache / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="index hash"):
        prepare.read_cache(cache)
    info["class_names"] = info["class_names"][::-1]
    # Corrupt the declared digest rather than overwriting a memory-mapped file.
    info["files"]["train.safetensors"] = "0" * 64
    (cache / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="mismatched"):
        prepare.read_cache(cache)
