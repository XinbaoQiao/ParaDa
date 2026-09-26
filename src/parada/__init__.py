"""ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning."""

from parada.federated import Client, Packet, aggregate, build_prior, fit
from parada.pipeline import adapt_episode, construct_classifier, predict
from parada.source import SourceMLP, train_or_load_source_model

__all__ = [
    "Client",
    "Packet",
    "aggregate",
    "build_prior",
    "fit",
    "SourceMLP",
    "adapt_episode",
    "construct_classifier",
    "predict",
    "train_or_load_source_model",
]
__version__ = "0.3.0"
