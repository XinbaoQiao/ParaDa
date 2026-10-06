import pytest
import torch
import torch.nn.functional as F
from parada.federated import build_prior, schedule


@pytest.mark.parametrize("k,total", [(1, 100), (2, 125), (5, 200), (10, 200)])
def test_schedule_endpoints(k, total):
    lr = schedule(k)
    assert len(lr) == total and lr[0] == 1e-5 and lr[1] == 0.002 and lr[-1] == 0
    assert all(lr[i] >= lr[i + 1] for i in range(1, total - 1))
    assert all(lr[i] != 0.002 for i in range(total // 5, total, total // 5))


def test_prior_normalizes_each_view_and_freezes_model():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    views = torch.tensor([[[10.0, 0.0], [0.0, 1.0], [0.0, 2.0]]])
    prior = build_prior(model, views)
    torch.testing.assert_close(prior, F.normalize(torch.tensor([[1.0, 2.0]]), dim=1))
    assert not model.training and not prior.requires_grad and not model.weight.requires_grad
