from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA

from .classical import CompressedMlpClassifier, DeepLenseResidualClassifier
from .config import CommonTestConfig, DataConfig, QuantumTestConfig
from .data import (
    LensingDatasetIndex,
    build_augmented_standardized_image_datasets,
    build_standardized_image_datasets,
)
from .evals import ClassificationMetrics, compute_multiclass_auc
from .quantum import VariationalQuantumClassifier
from .training import (
    TrainedClassifier,
    build_array_dataset,
    build_torch_dataloader,
    extract_embedding_features,
    predict_probabilities,
    train_torch_classifier,
)
from .typing_ import FloatArray, IntArray


@dataclass(frozen=True)
class CommonTestResult:
    metrics: ClassificationMetrics
    mean_losses: list[float]
    predicted_probabilities: FloatArray
    validation_labels: IntArray
    class_names: list[str]


@dataclass(frozen=True)
class QuantumTestResult:
    compressed_baseline_metrics: ClassificationMetrics
    vqc_metrics: ClassificationMetrics
    compressed_baseline_mean_losses: list[float]
    vqc_mean_losses: list[float]
    compressed_baseline_probabilities: FloatArray
    vqc_probabilities: FloatArray
    validation_labels: IntArray
    class_names: list[str]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _predict_image_probabilities(
        model: torch.nn.Module,
        dataset,
        *,
        batch_size: int,
        loader_workers: int,
        device: str,
) -> FloatArray:
    use_cuda = device.startswith("cuda")
    loader = build_torch_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        loader_workers=loader_workers,
    )
    model.to(device)
    model.eval()

    probability_batches: list[FloatArray] = []
    for batch_images, _ in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        if use_cuda:
            batch_images = batch_images.to(memory_format=torch.channels_last)
        logits = model(batch_images)
        probabilities = torch.softmax(logits, dim=1)
        probability_batches.append(probabilities.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False)


def _train_image_classifier(
        model: torch.nn.Module,
        *,
        train_dataset,
        validation_dataset,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        loader_workers: int,
        device: str,
        seed: int,
        use_cosine_schedule: bool,
) -> tuple[torch.nn.Module, TrainedClassifier, ClassificationMetrics, FloatArray]:
    if epochs <= 0:
        raise ValueError("experiments: epochs must be > 0")

    _set_seed(seed)
    use_cuda = device.startswith("cuda")
    train_loader = build_torch_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        loader_workers=loader_workers,
    )

    model = model.to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    if use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    mean_losses: list[float] = []
    best_macro_auc = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metrics: ClassificationMetrics | None = None
    best_probabilities: FloatArray | None = None

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            if use_cuda:
                batch_images = batch_images.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach().cpu().item())
            n_batches += 1

        mean_loss = loss_sum / max(n_batches, 1)
        mean_losses.append(mean_loss)

        probabilities = _predict_image_probabilities(
            model,
            validation_dataset,
            batch_size=batch_size,
            loader_workers=loader_workers,
            device=device,
        )
        metrics = compute_multiclass_auc(
            np.asarray(validation_dataset.split_index.labels, dtype=np.int64),
            probabilities,
            n_lensing_classes=len(validation_dataset.split_index.class_names),
        )
        print(
            f"training: epoch={epoch + 1}/{epochs} "
            f"mean_loss={mean_loss:.4f} macro_auc={metrics.macro_auc:.4f}"
        )
        if metrics.macro_auc > best_macro_auc:
            best_macro_auc = metrics.macro_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_probabilities = probabilities
        if scheduler is not None:
            scheduler.step()

    if best_state_dict is None or best_metrics is None or best_probabilities is None:
        raise RuntimeError("experiments: failed to produce validation metrics for the image classifier")

    model.load_state_dict(best_state_dict)
    return (
        model,
        TrainedClassifier(mean_losses=mean_losses),
        best_metrics,
        best_probabilities,
    )


def _stratified_subset(
        features: FloatArray,
        labels: IntArray,
        *,
        samples_per_class: int,
) -> tuple[FloatArray, IntArray]:
    subset_indices: list[int] = []
    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0][:samples_per_class]
        subset_indices.extend(class_indices.tolist())
    subset_array = np.asarray(subset_indices, dtype=np.int64)
    return features[subset_array], labels[subset_array]


def _fit_embedding_pca(
        train_embeddings: FloatArray,
        validation_embeddings: FloatArray,
        *,
        pca_components: int,
        seed: int,
) -> tuple[FloatArray, FloatArray]:
    pca_projector = PCA(n_components=pca_components, random_state=seed)
    reduced_train_features = pca_projector.fit_transform(train_embeddings)
    reduced_validation_features = pca_projector.transform(validation_embeddings)
    return (
        np.asarray(reduced_train_features, dtype=np.float32),
        np.asarray(reduced_validation_features, dtype=np.float32),
    )


def run_common_test(
        dataset_index: LensingDatasetIndex,
        *,
        data_config: DataConfig,
        test_config: CommonTestConfig,
        device: str = "cpu",
) -> CommonTestResult:
    train_dataset, validation_dataset = build_augmented_standardized_image_datasets(
        dataset_index,
        normalize=data_config.normalize,
        resized_image_size=test_config.resized_image_size,
    )

    n_lensing_classes = len(dataset_index.class_names)
    model = DeepLenseResidualClassifier(n_lensing_classes=n_lensing_classes)
    _, trained_model, metrics, predicted_probabilities = _train_image_classifier(
        model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=test_config.batch_size,
        learning_rate=test_config.learning_rate,
        weight_decay=test_config.weight_decay,
        epochs=test_config.epochs,
        loader_workers=data_config.loader_workers,
        device=device,
        seed=data_config.split_seed,
        use_cosine_schedule=True,
    )

    return CommonTestResult(
        metrics=metrics,
        mean_losses=trained_model.mean_losses,
        predicted_probabilities=predicted_probabilities,
        validation_labels=dataset_index.validation_split.labels,
        class_names=dataset_index.class_names,
    )


def run_quantum_test(
        dataset_index: LensingDatasetIndex,
        *,
        data_config: DataConfig,
        test_config: QuantumTestConfig,
        device: str = "cpu",
) -> QuantumTestResult:
    n_lensing_classes = len(dataset_index.class_names)

    common_test_config = CommonTestConfig()
    augmented_train_dataset, augmented_validation_dataset = build_augmented_standardized_image_datasets(
        dataset_index,
        normalize=data_config.normalize,
        resized_image_size=common_test_config.resized_image_size,
    )
    feature_backbone = DeepLenseResidualClassifier(n_lensing_classes=n_lensing_classes)
    feature_backbone, _, _, _ = _train_image_classifier(
        feature_backbone,
        train_dataset=augmented_train_dataset,
        validation_dataset=augmented_validation_dataset,
        batch_size=common_test_config.batch_size,
        learning_rate=common_test_config.learning_rate,
        weight_decay=common_test_config.weight_decay,
        epochs=common_test_config.epochs,
        loader_workers=data_config.loader_workers,
        device=device,
        seed=data_config.split_seed,
        use_cosine_schedule=True,
    )

    embedding_train_dataset, embedding_validation_dataset = build_standardized_image_datasets(
        dataset_index,
        normalize=data_config.normalize,
        resized_image_size=common_test_config.resized_image_size,
    )
    train_embeddings, train_labels = extract_embedding_features(
        feature_backbone,
        embedding_train_dataset,
        batch_size=test_config.embedding_batch_size,
        loader_workers=data_config.loader_workers,
        device=device,
    )
    validation_embeddings, validation_labels = extract_embedding_features(
        feature_backbone,
        embedding_validation_dataset,
        batch_size=test_config.embedding_batch_size,
        loader_workers=data_config.loader_workers,
        device=device,
    )

    reduced_train_features, reduced_validation_features = _fit_embedding_pca(
        train_embeddings,
        validation_embeddings,
        pca_components=test_config.pca_components,
        seed=data_config.split_seed,
    )

    compressed_baseline_model = CompressedMlpClassifier(
        input_dim=reduced_train_features.shape[1],
        n_lensing_classes=n_lensing_classes,
    )
    compressed_baseline_train_dataset = build_array_dataset(reduced_train_features, train_labels)
    compressed_baseline_validation_dataset = build_array_dataset(
        reduced_validation_features,
        validation_labels,
    )
    compressed_baseline_training = train_torch_classifier(
        compressed_baseline_model,
        compressed_baseline_train_dataset,
        batch_size=test_config.batch_size,
        learning_rate=test_config.compressed_learning_rate,
        weight_decay=test_config.weight_decay,
        epochs=test_config.compressed_epochs,
        loader_workers=data_config.loader_workers,
        device=device,
    )
    compressed_baseline_probabilities = predict_probabilities(
        compressed_baseline_model,
        compressed_baseline_validation_dataset,
        batch_size=test_config.batch_size,
        loader_workers=data_config.loader_workers,
        device=device,
    )
    compressed_baseline_metrics = compute_multiclass_auc(
        validation_labels,
        compressed_baseline_probabilities,
        n_lensing_classes=n_lensing_classes,
    )

    subset_train_features, subset_train_labels = _stratified_subset(
        reduced_train_features,
        train_labels,
        samples_per_class=test_config.quantum_train_samples_per_class,
    )
    vqc_model = VariationalQuantumClassifier(
        input_dim=reduced_train_features.shape[1],
        n_lensing_classes=n_lensing_classes,
        n_qubits=min(test_config.n_qubits, reduced_train_features.shape[1]),
        n_layers=test_config.n_layers,
        noise_strength=test_config.noise_strength,
    )
    vqc_train_dataset = build_array_dataset(subset_train_features, subset_train_labels)
    vqc_validation_dataset = build_array_dataset(reduced_validation_features, validation_labels)
    vqc_training = train_torch_classifier(
        vqc_model,
        vqc_train_dataset,
        batch_size=test_config.batch_size,
        learning_rate=test_config.quantum_learning_rate,
        weight_decay=test_config.weight_decay,
        epochs=test_config.quantum_epochs,
        loader_workers=data_config.loader_workers,
        device=device,
    )
    vqc_probabilities = predict_probabilities(
        vqc_model,
        vqc_validation_dataset,
        batch_size=test_config.batch_size,
        loader_workers=data_config.loader_workers,
        device=device,
    )
    vqc_metrics = compute_multiclass_auc(
        validation_labels,
        vqc_probabilities,
        n_lensing_classes=n_lensing_classes,
    )

    return QuantumTestResult(
        compressed_baseline_metrics=compressed_baseline_metrics,
        vqc_metrics=vqc_metrics,
        compressed_baseline_mean_losses=compressed_baseline_training.mean_losses,
        vqc_mean_losses=vqc_training.mean_losses,
        compressed_baseline_probabilities=compressed_baseline_probabilities,
        vqc_probabilities=vqc_probabilities,
        validation_labels=validation_labels,
        class_names=dataset_index.class_names,
    )
