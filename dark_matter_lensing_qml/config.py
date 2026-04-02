from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "dataset_common"


@dataclass(frozen=True)
class DataConfig:
    dataset_root: Path = _DEFAULT_DATASET_ROOT
    class_names: tuple[str, str, str] = ("no", "sphere", "vort")
    validation_fraction: float = 0.1
    split_seed: int = 0
    normalize: bool = True
    loader_workers: int = 0


@dataclass(frozen=True)
class CommonTestConfig:
    resized_image_size: int = 32
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 12


@dataclass(frozen=True)
class QuantumTestConfig:
    pca_components: int = 8
    embedding_batch_size: int = 128
    compressed_epochs: int = 8
    compressed_learning_rate: float = 1e-3
    batch_size: int = 16
    quantum_train_samples_per_class: int = 100
    n_qubits: int = 6
    n_layers: int = 1
    quantum_learning_rate: float = 5e-3
    quantum_epochs: int = 8
    weight_decay: float = 1e-4
    noise_strength: float = 0.0
