"""ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning."""

from parada.federated import PrototypePacket, aggregate_packets, build_prior, client_packet, fit
from parada.pipeline import adapt_episode, construct_classifier, predict
from parada.source import SourceMLP, train_or_load_source_model

__all__ = [
    "PrototypePacket",
    "client_packet",
    "aggregate_packets",
    "build_prior",
    "fit",
    "SourceMLP",
    "adapt_episode",
    "construct_classifier",
    "predict",
    "train_or_load_source_model",
]
__version__ = "0.5.0"
