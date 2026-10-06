import pytest
import torch
import torch.nn.functional as F
from parada.federated import PrototypePacket, aggregate_packets, client_packet, fit, fit_server


def test_count_weighting_preserves_client_means_without_renormalization():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -2.0], [1.0, 1.0], [-1.0, 1.0]])
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    owners = torch.tensor([0, 1, 1, 0, 0, 1])
    packets = [client_packet(i, x[owners == i], y[owners == i], 2) for i in range(10)]
    means, counts, audit = aggregate_packets(packets, 2, 2, 3)
    expected = torch.stack(
        [F.normalize(x.double(), dim=1)[y == c].mean(0) for c in range(2)]
    ).float()
    torch.testing.assert_close(means, expected, atol=1e-7, rtol=0)
    assert counts.tolist() == [3, 3]
    assert means[0].norm() < 1  # A class average is not normalized again.
    assert packets[0].means.dtype == torch.float32 and packets[0].counts.dtype == torch.int64
    assert not hasattr(packets[0], "features") and not hasattr(packets[0], "labels")
    assert packets[9].means.shape == (0, 2) and packets[9].counts.shape == (0,)
    assert audit["mean_upload_bytes"] == 4 * 2 * 4
    assert audit["count_upload_bytes"] == 4 * 8


def test_singleton_prototypes_equal_matched_individual_fit_and_reset():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        prior = torch.eye(2)
        x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        y = torch.arange(2)
        packets = [client_packet(i, x[i : i + 1], y[i : i + 1], 2) for i in range(10)]
        head, delta, audit = fit(packets, prior, 1)
        head_b, delta_b, server_b = fit_server(prior, x, y, torch.ones(2), 1)
        again = fit(packets, prior, 1)
        assert torch.equal(head, head_b) and torch.equal(delta, delta_b)
        assert torch.equal(head, again[0]) and torch.equal(delta, again[1])
        assert audit == again[2]
        assert audit["communication_rounds"] == 1
        assert audit["server_fit"]["steps"] == server_b["steps"] == 100
        assert audit["server_fit"]["initial_state_entries"] == 0
        assert audit["server_fit"]["final_step_loss"] < audit["server_fit"]["initial_loss"]
        assert torch.equal(prior, torch.eye(2))
        torch.testing.assert_close(head, F.normalize(prior + delta, dim=1))
    finally:
        torch.set_num_threads(previous)


def test_packet_rejects_client_identity_count_axis_and_precision_drift():
    x = torch.eye(2)
    y = torch.arange(2)
    packets = [client_packet(i, x[:0], y[:0], 2) for i in range(10)]
    packets[0] = client_packet(0, x, y, 2)
    with pytest.raises(ValueError, match="global per-class K"):
        aggregate_packets(packets, 2, 2, 2)
    with pytest.raises(ValueError, match="distinct client"):
        aggregate_packets([packets[0], packets[0], *packets[2:]], 2, 2, 1)
    for changed, message in [
        (PrototypePacket(0, (0, 0), torch.ones(2, dtype=torch.int64), x), "duplicate class"),
        (PrototypePacket(0, (0, 2), torch.ones(2, dtype=torch.int64), x), "shared axis"),
        (PrototypePacket(0, (0, 1), torch.ones(2, dtype=torch.int64), x.double()), "FP32"),
        (PrototypePacket(0, (0, 1), torch.ones(2), x), "count packet"),
    ]:
        with pytest.raises(ValueError, match=message):
            aggregate_packets([changed, *packets[1:]], 2, 2, 1)
