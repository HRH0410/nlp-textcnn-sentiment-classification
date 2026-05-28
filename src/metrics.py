from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(labels: Iterable[int], predictions: Iterable[int]) -> Dict[str, object]:
    y_true = list(labels)
    y_pred = list(predictions)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": np.asarray(confusion_matrix(y_true, y_pred)).tolist(),
    }
