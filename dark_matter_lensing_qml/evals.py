from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from .typing_ import FloatArray, IntArray


@dataclass(frozen=True)
class ClassificationMetrics:
    macro_auc: float
    micro_auc: float
    per_class_auc: tuple[float, ...]
    confusion: IntArray


@dataclass(frozen=True)
class RocCurveSet:
    false_positive_rates: list[FloatArray]
    true_positive_rates: list[FloatArray]
    auc_scores: tuple[float, ...]


def compute_multiclass_auc(
        labels: IntArray,
        predicted_probabilities: FloatArray,
        *,
        n_lensing_classes: int,
) -> ClassificationMetrics:
    if predicted_probabilities.ndim != 2:
        raise ValueError("evals: expected 2D probability matrix")
    if predicted_probabilities.shape[1] != n_lensing_classes:
        raise ValueError("evals: expected probabilities for the declared number of classes")

    labels_one_hot = label_binarize(labels, classes=np.arange(n_lensing_classes))
    if labels_one_hot.shape[1] != n_lensing_classes:
        raise ValueError("evals: failed to binarize labels for all classes")

    macro_auc = float(
        roc_auc_score(labels_one_hot, predicted_probabilities, average="macro", multi_class="ovr")
    )
    micro_auc = float(
        roc_auc_score(labels_one_hot, predicted_probabilities, average="micro", multi_class="ovr")
    )
    per_class_auc = tuple(
        float(roc_auc_score(labels_one_hot[:, class_id], predicted_probabilities[:, class_id]))
        for class_id in range(n_lensing_classes)
    )
    predicted_labels = np.argmax(predicted_probabilities, axis=1).astype(np.int64)
    confusion = confusion_matrix(labels, predicted_labels).astype(np.int64)

    return ClassificationMetrics(
        macro_auc=macro_auc,
        micro_auc=micro_auc,
        per_class_auc=per_class_auc,
        confusion=confusion,
    )


def one_vs_rest_roc_curves(
        labels: IntArray,
        predicted_probabilities: FloatArray,
        *,
        n_lensing_classes: int,
) -> RocCurveSet:
    labels_one_hot = label_binarize(labels, classes=np.arange(n_lensing_classes))
    false_positive_rates: list[FloatArray] = []
    true_positive_rates: list[FloatArray] = []
    auc_scores: list[float] = []

    for class_id in range(n_lensing_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, class_id], predicted_probabilities[:, class_id])
        false_positive_rates.append(np.asarray(fpr, dtype=np.float32))
        true_positive_rates.append(np.asarray(tpr, dtype=np.float32))
        auc_scores.append(
            float(roc_auc_score(labels_one_hot[:, class_id], predicted_probabilities[:, class_id]))
        )

    return RocCurveSet(
        false_positive_rates=false_positive_rates,
        true_positive_rates=true_positive_rates,
        auc_scores=tuple(auc_scores),
    )


def plot_roc_curves(
        roc_curves: RocCurveSet,
        *,
        class_names: list[str],
        title: str,
) -> None:
    plt.figure(figsize=(7, 5))
    for class_id, class_name in enumerate(class_names):
        plt.plot(
            roc_curves.false_positive_rates[class_id],
            roc_curves.true_positive_rates[class_id],
            label=f"{class_name} (AUC={roc_curves.auc_scores[class_id]:.3f})",
        )

    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()