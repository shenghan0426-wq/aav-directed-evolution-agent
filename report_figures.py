from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_rounds(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["strategy", "round", "mean_fitness", "best_fitness", "n"])
    fitness_col = "fitness" if "fitness" in df.columns else "target"
    if list(df.columns).count(fitness_col) > 1:
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return (
        df.groupby(["strategy", "round"])
        .agg(
            mean_fitness=(fitness_col, "mean"),
            best_fitness=(fitness_col, "max"),
            n=(fitness_col, "size"),
        )
        .reset_index()
        .sort_values(["strategy", "round"])
    )


def ensure_figure_dir(base_dir: str | Path = "artifacts/report_figures") -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
