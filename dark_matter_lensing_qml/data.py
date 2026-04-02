from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from .typing_ import FloatArray, IntArray, Tensor


@dataclass(frozen=True)
class LensingSplitIndex:
    split_name: str
    paths: list[Path]
    labels: IntArray
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.paths)


@dataclass(frozen=True)
class LensingDatasetIndex:
    train_split: LensingSplitIndex
    validation_split: LensingSplitIndex
    class_names: list[str]


def read_lensing_sample(path: str | Path) -> FloatArray:
    sample = np.load(path).astype(np.float32, copy=False)

    if sample.ndim != 3:
        raise ValueError(f"data: expected sample with 3 dimensions at {path}")
    if sample.shape[0] != 1:
        raise ValueError(f"data: expected leading channel dimension of size 1 at {path}")

    return sample


def build_lensing_split_index(
        split_root: str | Path,
        *,
        split_name: str,
        class_names: list[str],
) -> LensingSplitIndex:
    split_root = Path(split_root)

    if not split_root.exists():
        raise FileNotFoundError(f"data: missing split directory {split_root}")

    sample_paths: list[Path] = []
    label_ids: list[int] = []

    for class_id, class_name in enumerate(class_names):
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"data: missing class directory {class_dir}")

        class_paths = sorted(class_dir.glob("*.npy"))
        if not class_paths:
            raise ValueError(f"data: no .npy files found under class directory {class_dir}")

        sample_paths.extend(class_paths)
        label_ids.extend([class_id] * len(class_paths))

    return LensingSplitIndex(
        split_name=split_name,
        paths=sample_paths,
        labels=np.asarray(label_ids, dtype=np.int64),
        class_names=class_names,
    )


def build_lensing_split_from_items(
        *,
        split_name: str,
        sample_paths: list[Path],
        labels: IntArray,
        class_names: list[str],
) -> LensingSplitIndex:
    if len(sample_paths) != len(labels):
        raise ValueError("data: sample_paths and labels must have the same length")

    return LensingSplitIndex(
        split_name=split_name,
        paths=list(sample_paths),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
    )


def collect_lensing_samples(
        dataset_root: str | Path,
        *,
        class_names: list[str],
) -> tuple[list[Path], IntArray]:
    dataset_root = Path(dataset_root)

    sample_paths: list[Path] = []
    label_ids: list[int] = []
    for split_name in ["train", "val"]:
        split_index = build_lensing_split_index(
            dataset_root / split_name,
            split_name=split_name,
            class_names=class_names,
        )
        sample_paths.extend(split_index.paths)
        label_ids.extend(split_index.labels.tolist())

    return sample_paths, np.asarray(label_ids, dtype=np.int64)


def load_lensing_dataset_index(
        dataset_root: str | Path,
        *,
        class_names: list[str],
        validation_fraction: float = 0.1,
        split_seed: int = 0,
) -> LensingDatasetIndex:
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("data: validation_fraction must be in the open interval (0, 1)")

    # provided dataset is in 80:20 train/val folder split.
    # for solution, we rebuild a deterministic 90:10 split (that we are asked in test task)
    # from the combined sample pool
    all_paths, all_labels = collect_lensing_samples(
        dataset_root,
        class_names=class_names,
    )
    path_indices = np.arange(len(all_paths), dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=split_seed,
    )
    train_indices, validation_indices = next(splitter.split(path_indices, all_labels))

    train_split = build_lensing_split_from_items(
        split_name="train",
        sample_paths=[all_paths[int(index)] for index in train_indices],
        labels=all_labels[train_indices],
        class_names=class_names,
    )
    validation_split = build_lensing_split_from_items(
        split_name="val",
        sample_paths=[all_paths[int(index)] for index in validation_indices],
        labels=all_labels[validation_indices],
        class_names=class_names,
    )

    return LensingDatasetIndex(
        train_split=train_split,
        validation_split=validation_split,
        class_names=class_names,
    )


def count_samples_per_class(split_index: LensingSplitIndex) -> dict[str, int]:
    return {
        class_name: int(np.sum(split_index.labels == class_id))
        for class_id, class_name in enumerate(split_index.class_names)
    }


class LensingTorchDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
            self,
            split_index: LensingSplitIndex,
            *,
            normalize: bool,
            resized_image_size: int | None = None,
    ) -> None:
        self.split_index = split_index
        self.normalize = normalize
        self.resized_image_size = resized_image_size

    def __len__(self) -> int:
        return len(self.split_index)

    def __getitem__(self, item_index: int) -> tuple[Tensor, int]:
        sample_path = self.split_index.paths[item_index]
        sample = read_lensing_sample(sample_path)

        sample_tensor = torch.from_numpy(sample)
        if self.normalize:
            # This is more for safety and guarantees, as given dataset appears to be already normalized
            sample_tensor = sample_tensor.clamp(0.0, 1.0)

        if self.resized_image_size is not None:
            sample_tensor = F.interpolate(
                sample_tensor.unsqueeze(0),
                size=(self.resized_image_size, self.resized_image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        label = int(self.split_index.labels[item_index])
        return sample_tensor.to(torch.float32), label


def build_image_datasets(
        dataset_index: LensingDatasetIndex,
        *,
        normalize: bool,
        resized_image_size: int,
) -> tuple[LensingTorchDataset, LensingTorchDataset]:
    return (
        LensingTorchDataset(
            dataset_index.train_split,
            normalize=normalize,
            resized_image_size=resized_image_size,
        ),
        LensingTorchDataset(
            dataset_index.validation_split,
            normalize=normalize,
            resized_image_size=resized_image_size,
        ),
    )


class StandardizedImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
            self,
            base_dataset: LensingTorchDataset,
            *,
            mean_image: np.ndarray,
            std_image: np.ndarray,
    ) -> None:
        self.base_dataset = base_dataset
        self.split_index = base_dataset.split_index
        self.mean_image = torch.from_numpy(mean_image.astype(np.float32, copy=False))
        self.std_image = torch.from_numpy(std_image.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item_index: int) -> tuple[Tensor, int]:
        image, label = self.base_dataset[item_index]
        standardized_image = (image - self.mean_image) / self.std_image
        return standardized_image.to(torch.float32), label


class AugmentedImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
            self,
            base_dataset: Dataset[tuple[Tensor, int]],
            *,
            enable_rotations: bool = True,
            enable_flips: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.split_index = getattr(base_dataset, "split_index")
        self.enable_rotations = enable_rotations
        self.enable_flips = enable_flips

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item_index: int) -> tuple[Tensor, int]:
        image, label = self.base_dataset[item_index]
        if self.enable_flips and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
        if self.enable_flips and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(1,))
        if self.enable_rotations:
            rotation_k = int(torch.randint(0, 4, size=()).item())
            image = torch.rot90(image, k=rotation_k, dims=(1, 2))
        return image, label


def build_standardized_image_datasets(
        dataset_index: LensingDatasetIndex,
        *,
        normalize: bool,
        resized_image_size: int,
) -> tuple[StandardizedImageDataset, StandardizedImageDataset]:
    train_dataset, validation_dataset = build_image_datasets(
        dataset_index,
        normalize=normalize,
        resized_image_size=resized_image_size,
    )

    train_images = [
        train_dataset[item_index][0].numpy().astype(np.float32, copy=False)
        for item_index in range(len(train_dataset))
    ]
    stacked_train_images = np.stack(train_images, axis=0).astype(np.float32, copy=False)
    mean_image = stacked_train_images.mean(axis=0, dtype=np.float64).astype(np.float32)
    std_image = stacked_train_images.std(axis=0, dtype=np.float64).astype(np.float32)
    std_image = np.maximum(std_image, np.float32(1e-6))

    return (
        StandardizedImageDataset(
            train_dataset,
            mean_image=mean_image,
            std_image=std_image,
        ),
        StandardizedImageDataset(
            validation_dataset,
            mean_image=mean_image,
            std_image=std_image,
        ),
    )


def build_augmented_standardized_image_datasets(
        dataset_index: LensingDatasetIndex,
        *,
        normalize: bool,
        resized_image_size: int,
) -> tuple[AugmentedImageDataset, StandardizedImageDataset]:
    train_dataset, validation_dataset = build_standardized_image_datasets(
        dataset_index,
        normalize=normalize,
        resized_image_size=resized_image_size,
    )
    return AugmentedImageDataset(train_dataset), validation_dataset
