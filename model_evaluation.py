from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

try:
    from aav_baseline.llm_agent_core import FitnessMLP
except ModuleNotFoundError:
    from llm_agent_core import FitnessMLP


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_predictions(y_true, y_pred, top_k: int = 10) -> dict:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(true) == 0:
        raise ValueError("At least one prediction is required.")

    k = min(top_k, len(true))
    true_top = set(np.argsort(true)[-k:])
    pred_top = set(np.argsort(pred)[-k:])

    return {
        "n": int(len(true)),
        "mse": float(np.mean((true - pred) ** 2)),
        "pearson": _correlation(true, pred),
        "spearman": _correlation(_rank(true), _rank(pred)),
        "top_k": int(k),
        "top_k_hit_rate": float(len(true_top & pred_top) / k),
        "true_top_k_mean": float(np.mean(true[list(true_top)])),
        "predicted_top_k_true_mean": float(np.mean(true[list(pred_top)])),
    }


def one_hot_encode_regions(regions, width: int | None = None) -> np.ndarray:
    regions = list(regions)
    if not regions:
        return np.zeros((0, 0), dtype=np.float32)

    width = width or max(len(region) for region in regions)
    aa_to_index = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
    encoded = np.zeros((len(regions), width * len(AMINO_ACIDS)), dtype=np.float32)

    for row_index, region in enumerate(regions):
        for position, aa in enumerate(region[:width]):
            aa_index = aa_to_index.get(aa)
            if aa_index is not None:
                encoded[row_index, position * len(AMINO_ACIDS) + aa_index] = 1.0
    return encoded


def ridge_predictions(train_regions, train_targets, eval_regions, alpha: float = 10.0) -> np.ndarray:
    model = Ridge(alpha=alpha)
    train_regions = list(train_regions)
    eval_regions = list(eval_regions)
    width = max(max(map(len, train_regions)), max(map(len, eval_regions)))
    x_train = one_hot_encode_regions(train_regions, width=width)
    x_eval = one_hot_encode_regions(eval_regions, width=width)
    model.fit(x_train, np.asarray(train_targets, dtype=float))
    return model.predict(x_eval)


def ridge_baseline_predictions(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    mean_target = float(train_df["target"].mean())
    return np.full(len(eval_df), mean_target, dtype=float)


def predict_with_fitness_head(embeddings, checkpoint_path: str, batch_size: int = 4096) -> np.ndarray:
    x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    model = FitnessMLP()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            predictions.append(model(x[start : start + batch_size]).cpu().numpy())
    return np.concatenate(predictions)


def save_metrics(metrics: dict, path: str) -> pd.DataFrame:
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    return df
