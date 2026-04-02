from __future__ import annotations

from pathlib import Path

import numpy as np

from dark_matter_lensing_qml.data import (
    count_samples_per_class,
    load_lensing_dataset_index,
    read_lensing_sample,
)


def _write_fake_sample(path: Path, value: float) -> None:
    sample = np.full((1, 150, 150), value, dtype=np.float64)
    np.save(path, sample)


def test_load_dataset_index_and_counts(tmp_path: Path) -> None:
    for split_name in ["train", "val"]:
        for class_name in ["no", "sphere", "vort"]:
            class_dir = tmp_path / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(10):
                _write_fake_sample(class_dir / f"{sample_index}.npy", float(sample_index))

    dataset_index = load_lensing_dataset_index(
        tmp_path,
        class_names=["no", "sphere", "vort"],
    )

    assert len(dataset_index.train_split) == 54
    assert len(dataset_index.validation_split) == 6

    train_counts = count_samples_per_class(dataset_index.train_split)
    validation_counts = count_samples_per_class(dataset_index.validation_split)
    assert train_counts == {"no": 18, "sphere": 18, "vort": 18}
    assert validation_counts == {"no": 2, "sphere": 2, "vort": 2}


def test_read_lensing_sample_shape_and_dtype(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, np.ones((1, 150, 150), dtype=np.float64))

    sample = read_lensing_sample(sample_path)
    assert sample.shape == (1, 150, 150)
    assert sample.dtype == np.float32
