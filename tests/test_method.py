import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from parada import source
from parada.packet_io import load_packet, save_packet
from parada.pipeline import construct_classifier
from parada.sufficient_stats import client_statistics, solve_prior_ridge


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


def test_three_views_are_mapped_before_averaging():
    model, _, _, views = inputs()
    actual = source.mlp_classifier(model, views, torch.device("cpu"))
    with torch.no_grad():
        expected = F.normalize(
            torch.stack([F.normalize(model(views[:, i]), dim=1) for i in range(3)]).mean(dim=0),
            dim=1,
        )
    torch.testing.assert_close(actual, expected)


def test_k0_is_mlp_only_and_rejects_statistics():
    model, _, _, views = inputs()
    actual = construct_classifier(model, views, k=0)
    expected = source.mlp_classifier(model, views, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="K=0"):
        construct_classifier(model, views, k=0, packets=[])


def ridge_inputs():
    g = torch.Generator().manual_seed(94)
    prior = torch.randn(4, 7, generator=g)
    x = torch.randn(9, 7, generator=g)
    y = torch.tensor([0, 1, 1, 2, 3, 0, 2, 2, 3])
    return prior, x, y


def independent_central_solution(prior, x, y, regularization):
    # Augmented least squares independently checks the mean-loss scaling and prior anchor.
    x = F.normalize(x.double(), dim=1)
    w0 = F.normalize(prior.double(), dim=1)
    factor = (len(y) * regularization) ** 0.5
    design = torch.cat((x, factor * torch.eye(x.shape[1], dtype=torch.float64)))
    target = torch.cat((F.one_hot(y, len(prior)).double(), factor * w0.T))
    raw = torch.linalg.lstsq(design, target).solution.T
    return F.normalize(raw, dim=1).float(), (raw - w0).float()


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 2e-7), (torch.float32, 2e-6)])
def test_federated_matches_independent_central_ridge(dtype, tolerance):
    prior, x, y = ridge_inputs()
    # Unequal client sizes and an empty client must not induce client-mean weighting.
    packets = [
        client_statistics(i, x[a:b], y[a:b], classes=4, communication_dtype=dtype)
        for i, (a, b) in enumerate(((0, 2), (2, 9), (9, 9)))
    ]
    actual, residual, record = solve_prior_ridge(prior, packets)
    expected, expected_residual = independent_central_solution(prior, x, y, 0.01)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
    torch.testing.assert_close(residual, expected_residual, atol=tolerance, rtol=0)
    assert record["support_count"] == 9 and record["client_count"] == 3
    error = record["normal_equation_residual_l2"]
    assert isinstance(error, float) and error < 1e-10
    assert record["uploads_per_client"] == 1
    assert record["uploaded_support_features_or_labels"] is False
    torch.testing.assert_close(actual.norm(dim=1), torch.ones(4))


def test_sample_duplication_preserves_mean_loss_solution():
    prior, x, y = ridge_inputs()
    a = solve_prior_ridge(
        prior, [client_statistics(0, x, y, classes=4, communication_dtype=torch.float64)]
    )[0]
    b = solve_prior_ridge(
        prior,
        [
            client_statistics(
                0, x.repeat(2, 1), y.repeat(2), classes=4, communication_dtype=torch.float64
            )
        ],
    )[0]
    torch.testing.assert_close(a, b, atol=2e-7, rtol=0)


def test_support_adaptation_freezes_source_and_checks_count():
    model, _, _, views = inputs()
    before = source.state_hash(model)
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(10))
    packet = client_statistics(0, x, torch.arange(3), classes=3)
    actual = construct_classifier(model, views, k=1, packets=[packet])
    assert actual.shape == (3, 4) and source.state_hash(model) == before
    with pytest.raises(ValueError, match="total support count"):
        construct_classifier(model, views, k=2, packets=[packet])
    with pytest.raises(ValueError, match="requires client statistics"):
        construct_classifier(model, views, k=1)


def test_packet_serialization_and_no_raw_examples(tmp_path):
    from safetensors import safe_open

    _, x, y = ridge_inputs()
    packet = client_statistics(3, x, y, classes=4)
    path = tmp_path / "packet.safetensors"
    save_packet(path, packet)
    loaded = load_packet(path)
    assert loaded.client_id == 3 and loaded.count == 9
    torch.testing.assert_close(loaded.gram_upper, packet.gram_upper, rtol=0, atol=0)
    torch.testing.assert_close(loaded.cross, packet.cross, rtol=0, atol=0)
    assert packet.payload_bytes == 4 * (7 * 8 // 2 + 7 * 4) + 8
    with safe_open(str(path), framework="pt") as handle:
        assert set(handle.keys()) == {"gram_upper", "cross", "count"}
        assert handle.get_tensor("count").dtype == torch.int64
    with pytest.raises(source.SelectionError):
        save_packet(path, packet)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_regularization(value):
    prior, x, y = ridge_inputs()
    with pytest.raises(ValueError, match="regularization"):
        solve_prior_ridge(prior, [client_statistics(0, x, y, classes=4)], regularization=value)


def test_duplicate_empty_invalid_and_mixed_packets():
    from dataclasses import replace

    prior, x, y = ridge_inputs()
    packet = client_statistics(0, x, y, classes=4)
    with pytest.raises(ValueError, match="unique client"):
        solve_prior_ridge(prior, [packet, packet])
    with pytest.raises(ValueError, match="no labeled support"):
        solve_prior_ridge(prior, [client_statistics(0, x[:0], y[:0], classes=4)])
    for malformed in (
        replace(packet, count=-1),
        replace(packet, count=0),
        replace(packet, count=1.5),
        replace(packet, width=8),
        replace(packet, cross=packet.cross * float("nan")),
    ):
        with pytest.raises(ValueError):
            solve_prior_ridge(prior, [malformed])
    with pytest.raises(ValueError, match="same communication precision"):
        solve_prior_ridge(
            prior,
            [packet, client_statistics(1, x, y, classes=4, communication_dtype=torch.float64)],
        )
    with pytest.raises(ValueError, match="nonzero"):
        solve_prior_ridge(torch.zeros_like(prior), [packet])


def test_client_rejects_invalid_labels_and_features():
    _, x, y = ridge_inputs()
    for labels in (y.float(), y + 4, y[:-1]):
        with pytest.raises(ValueError):
            client_statistics(0, x, labels, classes=4)
    for features in (x * float("nan"), torch.zeros_like(x), x.long()):
        with pytest.raises(ValueError):
            client_statistics(0, features, y, classes=4)


def test_client_cli_and_packet_loader_reject_wrong_schema(tmp_path):
    from parada.cli import main
    from safetensors.torch import save_file

    _, x, y = ridge_inputs()
    local = tmp_path / "support.safetensors"
    output = tmp_path / "packet.safetensors"
    save_file({"support_features": x, "support_labels": y}, str(local))
    main(
        [
            "client-stats",
            "--support",
            str(local),
            "--classes",
            "4",
            "--client-id",
            "7",
            "--output",
            str(output),
        ]
    )
    assert load_packet(output).client_id == 7
    with pytest.raises(ValueError, match="packet format"):
        load_packet(local)
    with pytest.raises(SystemExit):
        main(["adapt", "--support", str(local)])


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
