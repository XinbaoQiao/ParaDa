import pytest
import torch
import torch.nn.functional as F
from parada.federated import Client, Packet, aggregate, build_prior, fit, schedule, tensor_digest


def test_weighting_and_empty_client_no_normalization():
    a = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    b = torch.tensor([[0.0, 6.0], [8.0, 0.0]])
    result = aggregate([Packet(0, 1, a), Packet(1, 3, b), Packet(2, 0, a * 1000)], (9, 2), a.shape)
    torch.testing.assert_close(result, (a + 3 * b) / 4)
    with pytest.raises(ValueError, match="class axis"):
        aggregate([Packet(0, 1, a)], (9,), a.shape)


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


def test_five_round_broadcast_moments_and_episode_reset():
    torch.set_num_threads(1)
    prior = torch.eye(5)

    def episode():
        clients = [
            Client(
                i,
                torch.eye(5)[(i + 1) % 5 : (i + 1) % 5 + 1] if i < 5 else torch.empty(0, 5),
                torch.tensor([i]) if i < 5 else torch.empty(0, dtype=torch.long),
                (8, 2, 5, 1, 7),
                prior,
                42,
                "cpu",
            )
            for i in range(10)
        ]
        return fit(clients, prior, (8, 2, 5, 1, 7), 1)

    w, delta, records = episode()
    w2, delta2, records2 = episode()
    torch.testing.assert_close(w, w2, rtol=0, atol=0)
    torch.testing.assert_close(delta, delta2, rtol=0, atol=0)
    assert records == records2 and len(records) == 5
    assert records[0]["broadcast_delta"] == tensor_digest(torch.zeros_like(prior))
    for i, record in enumerate(records):
        assert record["active_clients"] == 5 and record["support_count"] == 5
        assert {c["initial_delta"] for c in record["clients"] if c["count"]} == {
            record["broadcast_delta"]
        }
        assert record["download_delta_bytes"] == record["upload_delta_bytes"] == 5 * 25 * 4
        assert record["upload_count_bytes"] == 5 * 8
        assert all(c["optimizer_steps"] == (20 if c["count"] else 0) for c in record["clients"])
        if i:
            assert record["broadcast_delta"] == records[i - 1]["aggregate_delta"]
    torch.testing.assert_close(prior, torch.eye(5))


def test_reused_clients_reset_and_reject_wrong_prior():
    prior = torch.eye(2)
    clients = [
        Client(
            i,
            torch.tensor([[0.2, 1.0], [1.0, 0.2]]),
            torch.tensor([0, 1]),
            (7, 3),
            prior,
            42,
            "cpu",
        )
        for i in range(10)
    ]
    first = fit(clients, prior, (7, 3), 1)
    second = fit(clients, prior, (7, 3), 1)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="client prior"):
        fit(clients, prior.flip(0), (7, 3), 1)
    with pytest.raises(ValueError, match="class axis"):
        fit(clients, prior, (3, 7), 1)
