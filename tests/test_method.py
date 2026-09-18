from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from parada import correction, source
from parada.pipeline import construct_classifier, predict


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
        expected = F.normalize(torch.stack([
            F.normalize(model(views[:, i]), dim=1) for i in range(3)
        ]).mean(dim=0), dim=1)
    torch.testing.assert_close(actual, expected)


def test_source_dictionary_merges_duplicate_text_without_losing_alignment():
    text = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    visual = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    a, v, ids = correction.build_source_dictionary(text, visual, ("a", "b", "c"))
    assert len(ids) == len(set(ids)) == 2
    index = int(torch.where((a == torch.tensor([1., 0.])).all(dim=1))[0][0])
    torch.testing.assert_close(v[index], torch.tensor([2**-0.5, 2**-0.5], dtype=torch.float64))


def test_residual_correction_matches_independent_ridge_expression():
    _, text, visual, views = inputs()
    a, v, _ = correction.build_source_dictionary(text, visual, tuple(map(str, range(20))))
    actual = correction.residual_kernel_classifier(a, v, views)
    a, v = F.normalize(a, dim=1), F.normalize(v, dim=1)
    ac, vc = a - a.mean(0), v - v.mean(0)
    gram = ac.T @ ac
    ridge = .01 * torch.trace(gram) / a.shape[1]
    mapping = torch.linalg.solve(gram + ridge * torch.eye(a.shape[1]), ac.T @ vc)
    errors = v - (v.mean(0) + ac @ mapping)
    queries = F.normalize(views.double().reshape(-1, views.shape[-1]), dim=1)
    similarity = queries @ a.T
    neighbors = torch.argsort(similarity, descending=True, stable=True)[:, :16]
    weights = torch.softmax(torch.gather(similarity, 1, neighbors) / .1, dim=1)
    rows = v.mean(0) + (queries - a.mean(0)) @ mapping
    rows = rows + (weights[:, :, None] * errors[neighbors]).sum(1)
    expected = F.normalize(F.normalize(rows, dim=1).reshape(3, 3, 4).mean(1), dim=1).float()
    torch.testing.assert_close(actual, expected)


def test_k0_profiles_are_explicit_and_reject_support():
    model, text, visual, views = inputs()
    current = construct_classifier(model, text, visual, views, k=0)
    paper = construct_classifier(model, text, visual, views, k=0, rho_k0=.5)
    assert not torch.allclose(current, paper)
    torch.testing.assert_close(current.norm(dim=1), torch.ones(3))
    with pytest.raises(ValueError, match="K=0"):
        construct_classifier(model, text, visual, views, k=0, support_labels=torch.arange(3))


def test_support_update_preserves_mlp_and_separates_k0_coefficient():
    model, text, visual, views = inputs()
    support = F.normalize(torch.randn(3, 4, generator=torch.Generator().manual_seed(19)), dim=1)
    labels = torch.arange(3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    prior = source.mlp_classifier(model, views, torch.device("cpu"))
    first = construct_classifier(model, text, visual, views, k=1,
        support_features=support, support_labels=labels)
    second = construct_classifier(model, text, visual, views, k=1,
        support_features=support, support_labels=labels, rho_k0=.5)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)
    assert F.cross_entropy(predict(support, first), labels) < F.cross_entropy(
        predict(support, prior), labels
    )


def test_invalid_support_counts_and_schedule():
    model, text, visual, views = inputs()
    with pytest.raises(ValueError, match="exactly K"):
        construct_classifier(model, text, visual, views, k=1,
            support_features=torch.ones(2, 4), support_labels=torch.tensor([0, 1]))
    assert source.taskres_epochs(1) == 100
    assert source.taskres_epochs(5) == source.taskres_epochs(10) == 200
    schedule = source.taskres_schedule(1)
    assert schedule[0] == 1e-5 and schedule[1] == .002 and schedule[-1] == 0


def test_checkpoint_reuse_rejects_modified_source(tmp_path: Path, monkeypatch):
    # Two epochs test persistence only; this is not a benchmark result.
    monkeypatch.setattr(source, "MLP_EPOCHS", 2)
    _, text, visual, _ = inputs()
    trained, receipt = source.train_or_load_source_model(
        text, visual, seed=42, device=torch.device("cpu"), output_root=tmp_path)
    loaded, reused = source.train_or_load_source_model(
        text, visual, seed=42, device=torch.device("cpu"), output_root=tmp_path)
    assert receipt["execution"] == "trained" and reused["execution"] == "reused"
    assert source.state_hash(trained) == source.state_hash(loaded)
    with pytest.raises(source.SelectionError, match="receipt changed"):
        source.train_or_load_source_model(text + .1, visual, seed=42,
            device=torch.device("cpu"), output_root=tmp_path)


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
    code = '''
import sys
from pathlib import Path
import torch
from parada.source import write_once_safetensors, SelectionError
try:
    write_once_safetensors(Path(sys.argv[1]), {"value": torch.tensor([int(sys.argv[2])])}, {})
except (FileExistsError, SelectionError):
    sys.exit(3)
'''
    output = tmp_path / "race.safetensors"
    processes = [subprocess.Popen([sys.executable, "-c", code, str(output), str(i)],
        env=os.environ.copy()) for i in (1, 2)]
    assert sorted(p.wait(timeout=60) for p in processes) == [0, 3]
    from safetensors.torch import load_file
    assert int(load_file(str(output))["value"][0]) in (1, 2)
