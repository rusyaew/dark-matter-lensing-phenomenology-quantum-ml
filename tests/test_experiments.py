from __future__ import annotations

from pathlib import Path

import numpy as np

from dark_matter_lensing_qml import (
    CommonTestConfig,
    DataConfig,
    QuantumTestConfig,
    load_lensing_dataset_index,
    run_common_test,
    run_quantum_test,
)


def _write_fake_sample(path: Path, value: float) -> None:
    sample = np.full((1, 150, 150), value, dtype=np.float64)
    np.save(path, sample)


def _build_fake_dataset(tmp_path: Path) -> Path:
    for split_name in ["train", "val"]:
        for class_id, class_name in enumerate(["no", "sphere", "vort"]):
            class_dir = tmp_path / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(10):
                _write_fake_sample(
                    class_dir / f"{sample_index}.npy",
                    float(class_id * 10 + sample_index),
                )
    return tmp_path


def test_run_common_and_quantum_experiments_smoke(tmp_path: Path) -> None:
    dataset_root = _build_fake_dataset(tmp_path)
    data_config = DataConfig(
        dataset_root=dataset_root,
        loader_workers=0,
    )
    dataset_index = load_lensing_dataset_index(
        dataset_root,
        class_names=list(data_config.class_names),
    )

    common_result = run_common_test(
        dataset_index,
        data_config=data_config,
        test_config=CommonTestConfig(epochs=1),
        device="cpu",
    )
    assert len(common_result.mean_losses) == 1
    assert common_result.predicted_probabilities.shape == (len(dataset_index.validation_split), 3)

    quantum_result = run_quantum_test(
        dataset_index,
        data_config=data_config,
        test_config=QuantumTestConfig(
            pca_components=4,
            embedding_batch_size=16,
            compressed_epochs=1,
            quantum_train_samples_per_class=1,
            quantum_epochs=1,
            n_qubits=2,
            n_layers=1,
        ),
        device="cpu",
    )
    assert len(quantum_result.compressed_baseline_mean_losses) == 1
    assert len(quantum_result.vqc_mean_losses) == 1
    assert quantum_result.compressed_baseline_probabilities.shape == (
        len(dataset_index.validation_split),
        3,
    )
    assert quantum_result.vqc_probabilities.shape == (len(dataset_index.validation_split), 3)
