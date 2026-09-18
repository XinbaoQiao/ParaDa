"""ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning."""

from parada.pipeline import construct_classifier, predict
from parada.source import SourceMLP, train_or_load_source_model

__all__ = ["SourceMLP", "construct_classifier", "predict", "train_or_load_source_model"]
__version__ = "0.1.0"
