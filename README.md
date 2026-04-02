# dark-matter-lensing-phenomenology-quantum-ml

A compact research-style implementation for strong-lensing dark-matter phenomenology.

It is written in typed Python / PyTorch, with lazy loading for the `.npy` lensing dataset, deterministic split rebuilding, explicit experiment runners for the final common baseline and the project-specific hybrid quantum-classical path, and a small smoke-test suite. The Unicode diagrams for easier review are from asciip.dev.

This repository contains my solutions for:

- `notebooks/common_test_i.ipynb`
- `notebooks/quantum_ml_iii.ipynb`

The notebooks are the main submission artifacts. The Python package under `dark_matter_lensing_qml/` provides the shared code they use for dataset loading, training, evaluation, and the final common / quantum model paths.

The repo also includes trained weight artifacts under `submission_weights/`:

- `submission_weights/common_test_i_residual_cnn.pt`
- `submission_weights/quantum_ml_iii_bundle.pt`

## Dataset layout

The code expects the provided `.npy` dataset under:

```text
datasets/
  dataset_common/
    train/
      no/
      sphere/
      vort/
    val/
      no/
      sphere/
      vort/
```

The on-disk folders are currently arranged as `train/` and `val/`, but the code rebuilds a deterministic stratified `90:10` split from the combined sample pool, because the test brief asks for validation metrics on a `90:10` train-test split.

## What is in each notebook

### Common Test I

`notebooks/common_test_i.ipynb`:

- inspects the dataset and the rebuilt split,
- compares a few classical image models,
- explains why the final `32x32` residual CNN path was selected,
- trains the selected model,
- reports ROC curves and AUC metrics on the validation split.

### Quantum Test III

`notebooks/quantum_ml_iii.ipynb`:

- checks whether raw compressed image features are informative,
- shows why the original raw `16x16 -> PCA` path was too weak,
- trains a residual backbone and extracts learned morphology embeddings,
- compares a compact classical head and a variational quantum classifier on the same compressed embedding path,
- reports ROC curves and AUC metrics on the validation split.

## Architecture diagrams
```text
╭────────────────────────────────────────────────────────────────────────────╮
│ typing_.py                                                                 │
├────────────────────────────────────────────────────────────────────────────┤
│ FloatArray = NDArray[np.float32]                                           │
│ IntArray   = NDArray[np.int64]                                             │
│ Tensor     = torch.Tensor                                                  │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ config.py                                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│ DataConfig(                                                                │
│   dataset_root: Path = datasets/dataset_common,                            │
│   class_names: tuple[str, str, str] = ("no", "sphere", "vort"),            │
│   validation_fraction: float = 0.1,                                        │
│   split_seed: int = 0,                                                     │
│   normalize: bool = True,                                                  │
│   loader_workers: int = 0                                                  │
│ )                                                                          │
│                                                                            │
│ CommonTestConfig(                                                          │
│   resized_image_size: int = 32,                                            │
│   batch_size: int = 32,                                                    │
│   learning_rate: float = 2e-4,                                             │
│   weight_decay: float = 1e-4,                                              │
│   epochs: int = 12                                                         │
│ )                                                                          │
│                                                                            │
│ QuantumTestConfig(                                                         │
│   pca_components: int = 8,                                                 │
│   embedding_batch_size: int = 128,                                         │
│   compressed_epochs: int = 8,                                              │
│   compressed_learning_rate: float = 1e-3,                                  │
│   batch_size: int = 16,                                                    │
│   quantum_train_samples_per_class: int = 100,                              │
│   n_qubits: int = 6,                                                       │
│   n_layers: int = 1,                                                       │
│   quantum_learning_rate: float = 5e-3,                                     │
│   quantum_epochs: int = 8,                                                 │
│   weight_decay: float = 1e-4,                                              │
│   noise_strength: float = 0.0                                              │
│ )                                                                          │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ data.py                                                                    │
├────────────────────────────────────────────────────────────────────────────┤
│ LensingSplitIndex(                                                         │
│   split_name: str,                                                         │
│   paths: list[Path],                                                       │
│   labels: IntArray,                                                        │
│   class_names: list[str]                                                   │
│ )                                                                          │
│   __len__() -> int                                                         │
│                                                                            │
│ LensingDatasetIndex(                                                       │
│   train_split: LensingSplitIndex,                                          │
│   validation_split: LensingSplitIndex,                                     │
│   class_names: list[str]                                                   │
│ )                                                                          │
│                                                                            │
│ read_lensing_sample(path: str | Path) -> FloatArray                        │
│ build_lensing_split_index(...) -> LensingSplitIndex                        │
│ build_lensing_split_from_items(...) -> LensingSplitIndex                   │
│ collect_lensing_samples(...) -> tuple[list[Path], IntArray]                │
│ load_lensing_dataset_index(...) -> LensingDatasetIndex                     │
│ count_samples_per_class(split_index: LensingSplitIndex)                    │
│   -> dict[str, int]                                                        │
│                                                                            │
│ LensingTorchDataset(split_index: LensingSplitIndex,                        │
│                     normalize: bool,                                       │
│                     resized_image_size: int | None = None)                 │
│   __len__() -> int                                                         │
│   __getitem__(item_index: int) -> tuple[Tensor, int]                       │
│                                                                            │
│ build_image_datasets(...)                                                  │
│ build_standardized_image_datasets(...)                                     │
│ build_augmented_standardized_image_datasets(...)                           │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ classical.py                                                               │
├────────────────────────────────────────────────────────────────────────────┤
│ CompressedMlpClassifier(                                                   │
│   input_dim: int,                                                          │
│   n_lensing_classes: int,                                                  │
│   hidden_dim: int = 64                                                     │
│ )                                                                          │
│   forward(features: torch.Tensor) -> torch.Tensor                          │
│                                                                            │
│ ResidualBlock(                                                             │
│   in_channels: int,                                                        │
│   out_channels: int,                                                       │
│   stride: int = 1                                                          │
│ )                                                                          │
│   forward(images: torch.Tensor) -> torch.Tensor                            │
│                                                                            │
│ DeepLenseResidualClassifier(                                               │
│   n_lensing_classes: int,                                                  │
│   embedding_dim: int = 128                                                 │
│ )                                                                          │
│   forward_features(images: torch.Tensor) -> torch.Tensor                   │
│   forward(images: torch.Tensor) -> torch.Tensor                            │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ quantum.py                                                                 │
├────────────────────────────────────────────────────────────────────────────┤
│ apply_angle_encoding(angle_features: torch.Tensor) -> None                 │
│ apply_variational_layer(layer_weights: torch.Tensor, *,                    │
│                         n_qubits: int,                                     │
│                         noise_strength: float) -> None                     │
│ build_variational_quantum_circuit(*,                                       │
│   n_qubits: int,                                                           │
│   n_layers: int,                                                           │
│   noise_strength: float                                                    │
│ ) -> Callable                                                              │
│                                                                            │
│ VariationalQuantumClassifier(                                              │
│   input_dim: int,                                                          │
│   n_lensing_classes: int,                                                  │
│   n_qubits: int,                                                           │
│   n_layers: int,                                                           │
│   noise_strength: float                                                    │
│ )                                                                          │
│   forward(compressed_features: torch.Tensor) -> torch.Tensor               │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ evals.py                                                                   │
├────────────────────────────────────────────────────────────────────────────┤
│ ClassificationMetrics(                                                     │
│   macro_auc: float,                                                        │
│   micro_auc: float,                                                        │
│   per_class_auc: tuple[float, ...],                                        │
│   confusion: IntArray                                                      │
│ )                                                                          │
│                                                                            │
│ RocCurveSet(                                                               │
│   false_positive_rates: list[FloatArray],                                  │
│   true_positive_rates: list[FloatArray],                                   │
│   auc_scores: tuple[float, ...]                                            │
│ )                                                                          │
│                                                                            │
│ compute_multiclass_auc(labels: IntArray,                                   │
│                        predicted_probabilities: FloatArray, *,             │
│                        n_lensing_classes: int)                             │
│   -> ClassificationMetrics                                                 │
│ one_vs_rest_roc_curves(labels: IntArray,                                   │
│                        predicted_probabilities: FloatArray, *,             │
│                        n_lensing_classes: int)                             │
│   -> RocCurveSet                                                           │
│ plot_roc_curves(roc_curves: RocCurveSet, *,                                │
│                 class_names: list[str],                                    │
│                 title: str) -> None                                        │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ training.py                                                                │
├────────────────────────────────────────────────────────────────────────────┤
│ TrainedClassifier(mean_losses: list[float])                                │
│                                                                            │
│ build_array_dataset(features: FloatArray, labels: IntArray)                │
│   -> TensorDataset                                                         │
│ build_torch_dataloader(dataset: Dataset, *,                                │
│                        batch_size: int,                                    │
│                        shuffle: bool,                                      │
│                        loader_workers: int)                                │
│   -> DataLoader                                                            │
│ train_torch_classifier(model: nn.Module, train_dataset: Dataset, *,        │
│                        batch_size: int,                                    │
│                        learning_rate: float,                               │
│                        weight_decay: float,                                │
│                        epochs: int,                                        │
│                        loader_workers: int,                                │
│                        device: str)                                        │
│   -> TrainedClassifier                                                     │
│ predict_probabilities(model: nn.Module, dataset: Dataset, *,               │
│                       batch_size: int,                                     │
│                       loader_workers: int,                                 │
│                       device: str)                                         │
│   -> FloatArray                                                            │
│ extract_embedding_features(model: nn.Module, dataset: Dataset, *,          │
│                            batch_size: int,                                │
│                            loader_workers: int,                            │
│                            device: str)                                    │
│   -> tuple[FloatArray, IntArray]                                           │
╰────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────╮
│ experiments.py                                                             │
├────────────────────────────────────────────────────────────────────────────┤
│ CommonTestResult(                                                          │
│   metrics: ClassificationMetrics,                                          │
│   mean_losses: list[float],                                                │
│   predicted_probabilities: FloatArray,                                     │
│   validation_labels: IntArray,                                             │
│   class_names: list[str]                                                   │
│ )                                                                          │
│                                                                            │
│ QuantumTestResult(                                                         │
│   compressed_baseline_metrics: ClassificationMetrics,                      │
│   vqc_metrics: ClassificationMetrics,                                      │
│   compressed_baseline_mean_losses: list[float],                            │
│   vqc_mean_losses: list[float],                                            │
│   compressed_baseline_probabilities: FloatArray,                           │
│   vqc_probabilities: FloatArray,                                           │
│   validation_labels: IntArray,                                             │
│   class_names: list[str]                                                   │
│ )                                                                          │
│                                                                            │
│ run_common_test(...) -> CommonTestResult                                   │
│ run_quantum_test(...) -> QuantumTestResult                                 │
╰────────────────────────────────────────────────────────────────────────────╯
```

## Running

For quick package checks:

```text
./.venv/bin/pytest -q
```
