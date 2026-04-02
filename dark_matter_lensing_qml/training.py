from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .typing_ import FloatArray, IntArray


@dataclass(frozen=True)
class TrainedClassifier:
    mean_losses: list[float]


def build_array_dataset(features: FloatArray, labels: IntArray) -> TensorDataset:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False))
    return TensorDataset(feature_tensor, label_tensor)


def build_torch_dataloader(
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        loader_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=loader_workers > 0,
    )


def train_torch_classifier(
        model: nn.Module,
        train_dataset: Dataset,
        *,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        loader_workers: int,
        device: str,
) -> TrainedClassifier:
    if epochs <= 0:
        raise ValueError("training: epochs must be > 0")

    train_loader = build_torch_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        loader_workers=loader_workers,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    mean_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu().item())
            n_batches += 1

        mean_loss = loss_sum / max(n_batches, 1)
        mean_losses.append(mean_loss)
        print(f"training: epoch={epoch + 1}/{epochs} mean_loss={mean_loss:.4f}")

    return TrainedClassifier(mean_losses=mean_losses)


@torch.no_grad()
def predict_probabilities(
        model: nn.Module,
        dataset: Dataset,
        *,
        batch_size: int,
        loader_workers: int,
        device: str,
) -> FloatArray:
    loader = build_torch_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        loader_workers=loader_workers,
    )

    model.to(device)
    model.eval()

    probability_batches: list[FloatArray] = []

    for batch_features, _ in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        logits = model(batch_features)
        probabilities = torch.softmax(logits, dim=1)
        probability_batches.append(probabilities.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def extract_embedding_features(
        model: nn.Module,
        dataset: Dataset,
        *,
        batch_size: int,
        loader_workers: int,
        device: str,
) -> tuple[FloatArray, IntArray]:
    if not hasattr(model, "forward_features"):
        raise AttributeError("training: model must define forward_features(...) for embedding extraction")

    loader = build_torch_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        loader_workers=loader_workers,
    )
    model.to(device)
    model.eval()

    feature_batches: list[FloatArray] = []
    label_batches: list[IntArray] = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        embeddings = model.forward_features(batch_features)
        feature_batches.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
        label_batches.append(batch_labels.numpy().astype(np.int64, copy=False))

    return (
        np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False),
        np.concatenate(label_batches, axis=0).astype(np.int64, copy=False),
    )
