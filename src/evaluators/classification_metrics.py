import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import List, Tuple
from src.utils.schema import MetricSchema


def compute_results(
    y_true: List[int], y_pred: List[int], labels: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Metrics
    acc: float = accuracy_score(y_true, y_pred)
    prec: float = precision_score(y_true, y_pred, average="weighted")
    rec: float = recall_score(y_true, y_pred, average="weighted")
    f1: float = f1_score(y_true, y_pred, average="weighted")

    metrics_df: pd.DataFrame = pd.DataFrame(
        {
            MetricSchema.ACCURACY: [acc],
            MetricSchema.PRECISION: [prec],
            MetricSchema.RECALL: [rec],
            MetricSchema.F1_SCORE: [f1],
        }
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df: pd.DataFrame = pd.DataFrame(cm, index=labels, columns=labels)

    return metrics_df, cm_df
