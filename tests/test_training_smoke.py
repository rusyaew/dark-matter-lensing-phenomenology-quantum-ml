from __future__ import annotations

import numpy as np
import pytest
import torch

from dark_matter_lensing_qml.classical import CompressedMlpClassifier
from dark_matter_lensing_qml.quantum import VariationalQuantumClassifier
from dark_matter_lensing_qml.training import (
    build_array_dataset,
    predict_probabilities,
    train_torch_classifier,
)


def _make_small_multiclass_arrays(
        *,
        n_train_samples: int,
        n_validation_samples: int,
        n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_train_samples % 3 != 0 or n_validation_samples % 3 != 0:
        raise ValueError("sample counts must be divisible by 3")

    rng = np.random.default_rng(0)
    train_features = rng.normal(size=(n_train_samples, n_features)).astype(np.float32)
    train_labels = np.asarray([0, 1, 2] * (n_train_samples // 3), dtype=np.int64)
    validation_features = rng.normal(size=(n_validation_samples, n_features)).astype(np.float32)
    validation_labels = np.asarray([0, 1, 2] * (n_validation_samples // 3), dtype=np.int64)
    return train_features, train_labels, validation_features, validation_labels


def test_train_and_predict_smoke() -> None:
    train_features, train_labels, validation_features, validation_labels = _make_small_multiclass_arrays(
        n_train_samples=18,
        n_validation_samples=9,
        n_features=6,
    )

    model = CompressedMlpClassifier(
        input_dim=6,
        n_lensing_classes=3,
        hidden_dim=16,
    )

    train_dataset = build_array_dataset(train_features, train_labels)
    validation_dataset = build_array_dataset(validation_features, validation_labels)

    trained_model = train_torch_classifier(
        model,
        train_dataset,
        batch_size=6,
        learning_rate=1e-2,
        weight_decay=0.0,
        epochs=2,
        loader_workers=0,
        device="cpu",
    )
    predicted_probabilities = predict_probabilities(
        model,
        validation_dataset,
        batch_size=3,
        loader_workers=0,
        device="cpu",
    )

    assert len(trained_model.mean_losses) == 2
    assert predicted_probabilities.shape == (9, 3)
    assert np.all(np.isfinite(predicted_probabilities))


def test_train_and_predict_quantum_smoke_cpu() -> None:
    train_features, train_labels, validation_features, validation_labels = _make_small_multiclass_arrays(
        n_train_samples=12,
        n_validation_samples=6,
        n_features=8,
    )

    model = VariationalQuantumClassifier(
        input_dim=8,
        n_lensing_classes=3,
        n_qubits=4,
        n_layers=1,
        noise_strength=0.0,
    )

    train_dataset = build_array_dataset(train_features, train_labels)
    validation_dataset = build_array_dataset(validation_features, validation_labels)

    trained_model = train_torch_classifier(
        model,
        train_dataset,
        batch_size=3,
        learning_rate=1e-2,
        weight_decay=0.0,
        epochs=1,
        loader_workers=0,
        device="cpu",
    )
    predicted_probabilities = predict_probabilities(
        model,
        validation_dataset,
        batch_size=3,
        loader_workers=0,
        device="cpu",
    )

    assert len(trained_model.mean_losses) == 1
    assert predicted_probabilities.shape == (6, 3)
    assert np.all(np.isfinite(predicted_probabilities))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_train_and_predict_quantum_smoke_cuda() -> None:
    train_features, train_labels, validation_features, validation_labels = _make_small_multiclass_arrays(
        n_train_samples=12,
        n_validation_samples=6,
        n_features=8,
    )

    model = VariationalQuantumClassifier(
        input_dim=8,
        n_lensing_classes=3,
        n_qubits=4,
        n_layers=1,
        noise_strength=0.0,
    )

    train_dataset = build_array_dataset(train_features, train_labels)
    validation_dataset = build_array_dataset(validation_features, validation_labels)

    trained_model = train_torch_classifier(
        model,
        train_dataset,
        batch_size=3,
        learning_rate=1e-2,
        weight_decay=0.0,
        epochs=1,
        loader_workers=0,
        device="cuda",
    )
    predicted_probabilities = predict_probabilities(
        model,
        validation_dataset,
        batch_size=3,
        loader_workers=0,
        device="cuda",
    )

    assert len(trained_model.mean_losses) == 1
    assert predicted_probabilities.shape == (6, 3)
    assert np.all(np.isfinite(predicted_probabilities))
