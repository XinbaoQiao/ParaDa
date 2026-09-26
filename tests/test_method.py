import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from parada import source
from parada.federated import build_prior
from parada.pipeline import adapt_episode, construct_classifier


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs():
    generator = torch.Generator().manual_seed(83)
    text = torch.randn(20, 5, generator=generator)
    visual = torch.randn(20, 4, generator=generator)
    views = torch.randn(3, 3, 5, generator=generator)
    torch.manual_seed(17)
    model = source.SourceMLP(5, 4).eval()
    return model, text, visual, views


def support_inputs():
    g = torch.Generator().manual_seed(10)
    x = torch.randn(3, 4, generator=g)
    y = torch.arange(3)
    return [(x[:2], y[:2]), (x[2:], y[2:])] + [(x[:0], y[:0])] * 8


def test_k0_is_mlp_only_and_rejects_support():
    model, _, _, views = inputs()
    actual = construct_classifier(model, views, k=0)
    expected = build_prior(model, views)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="K=0"):
        construct_classifier(model, views, k=0, supports=[])


def test_five_round_pipeline_freezes_source_and_global_k():
    model, _, _, views = inputs()
    before = source.state_hash(model)
    supports = support_inputs()
    weights, delta, rounds = adapt_episode(model, views, k=1, supports=supports)
    assert source.state_hash(model) == before
    assert len(rounds) == 5 and all(r["support_count"] == 3 for r in rounds)
    torch.testing.assert_close(weights, F.normalize(build_prior(model, views) + delta, dim=1))
    with pytest.raises(ValueError, match="exactly K"):
        construct_classifier(model, views, k=2, supports=supports)
    # Correct total but wrong per-class allocation must also be rejected.
    wrong = [(x, torch.zeros_like(y)) for x, y in supports]
    with pytest.raises(ValueError, match="exactly K"):
        construct_classifier(model, views, k=1, supports=wrong)
    with pytest.raises(ValueError, match="ten client"):
        construct_classifier(model, views, k=1, supports=supports[:2])


def test_support_schema_rejected_before_training():
    model, _, _, views = inputs()
    supports = support_inputs()
    x, y = supports[0]
    for bad in (
        (x, y.float()),
        (x, y + 3),
        (x, y[:1]),
        (x * float("nan"), y),
        (x * 0, y),
        (x.long(), y),
    ):
        with pytest.raises(ValueError):
            construct_classifier(model, views, k=1, supports=[bad, *supports[1:]])


def test_checkpoint_reuse_rejects_modified_source(tmp_path: Path, monkeypatch):
    # Two epochs test persistence only; this is not a benchmark result.
    monkeypatch.setattr(source, "MLP_EPOCHS", 2)
    _, text, visual, _ = inputs()
    trained, receipt = source.train_or_load_source_model(
        text, visual, seed=42, device=torch.device("cpu"), output_root=tmp_path
    )
    loaded, reused = source.train_or_load_source_model(
        text, visual, seed=42, device=torch.device("cpu"), output_root=tmp_path
    )
    assert receipt["execution"] == "trained" and reused["execution"] == "reused"
    assert source.state_hash(trained) == source.state_hash(loaded)
    with pytest.raises(source.SelectionError, match="receipt changed"):
        source.train_or_load_source_model(
            text + 0.1, visual, seed=42, device=torch.device("cpu"), output_root=tmp_path
        )


def test_cli_rejects_existing_and_dangling_output(tmp_path: Path):
    from parada.cli import main

    output = tmp_path / "output.safetensors"
    output.write_bytes(b"preserve this output")
    args = ["predict", "--features", "unused", "--classifier", "unused", "--output", str(output)]
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == b"preserve this output"
    link = tmp_path / "dangling.safetensors"
    target = tmp_path / "not-created.safetensors"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic link creation unavailable")
    with pytest.raises(FileExistsError):
        main([*args[:-1], str(link)])
    assert not target.exists()


def test_concurrent_output_writers_never_replace_first_artifact(tmp_path: Path):
    code = """
import sys
from pathlib import Path
import torch
from parada.source import write_once_safetensors, SelectionError
try:
    write_once_safetensors(Path(sys.argv[1]), {"value": torch.tensor([int(sys.argv[2])])}, {})
except (FileExistsError, SelectionError):
    sys.exit(3)
"""
    output = tmp_path / "race.safetensors"
    processes = [
        subprocess.Popen([sys.executable, "-c", code, str(output), str(i)], env=os.environ.copy())
        for i in (1, 2)
    ]
    assert sorted(p.wait(timeout=60) for p in processes) == [0, 3]
    from safetensors.torch import load_file

    assert int(load_file(str(output))["value"][0]) in (1, 2)
