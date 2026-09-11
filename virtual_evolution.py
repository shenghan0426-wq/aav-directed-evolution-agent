from __future__ import annotations

import random
import re

import pandas as pd


def select_random_candidates(candidate_ids, top_k: int = 10, seed: int | None = None) -> list:
    ids = list(candidate_ids)
    rng = random.Random(seed)
    return rng.sample(ids, k=min(top_k, len(ids)))


def select_model_top_candidates(candidate_df: pd.DataFrame, top_k: int = 10, score_col: str = "predicted_fitness") -> list[str]:
    return (
        candidate_df.sort_values(score_col, ascending=False)
        .head(top_k)["candidate_id"]
        .astype(str)
        .tolist()
    )


def evaluate_selected(candidate_df: pd.DataFrame, selected_ids: list[str], fitness_col: str = "target") -> pd.DataFrame:
    selected = candidate_df[candidate_df["candidate_id"].isin(selected_ids)].copy()
    rank_map = {candidate_id: rank for rank, candidate_id in enumerate(selected_ids, start=1)}
    selected["selection_rank"] = selected["candidate_id"].map(rank_map)
    selected = selected.sort_values("selection_rank")
    if fitness_col in selected.columns:
        selected["fitness"] = selected[fitness_col]
    return selected


def simulate_rounds(candidate_df: pd.DataFrame, strategies: dict[str, callable], rounds: int = 3, top_k: int = 10) -> pd.DataFrame:
    remaining = candidate_df.copy()
    rows = []
    for round_index in range(1, rounds + 1):
        if remaining.empty:
            break
        for strategy_name, strategy in strategies.items():
            selected_ids = strategy(remaining, round_index, top_k)
            selected = evaluate_selected(remaining, selected_ids)
            for _, row in selected.iterrows():
                record = row.to_dict()
                record["strategy"] = strategy_name
                record["round"] = round_index
                rows.append(record)
        used_ids = {row["candidate_id"] for row in rows if row["round"] == round_index}
        remaining = remaining[~remaining["candidate_id"].isin(used_ids)].copy()
    return pd.DataFrame(rows)


def simulate_iterative_evolution(
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    strategies: dict[str, callable],
    rounds: int = 3,
    top_k: int = 10,
    evaluator_col: str = "target",
) -> pd.DataFrame:
    rows = []

    for strategy_name, strategy in strategies.items():
        observed = observed_df.copy()
        remaining = candidate_df.copy()
        initial_best = float(observed[evaluator_col].max()) if evaluator_col in observed.columns and not observed.empty else None
        initial_mean = float(observed[evaluator_col].mean()) if evaluator_col in observed.columns and not observed.empty else None

        rows.append(
            {
                "strategy": strategy_name,
                "round": 0,
                "candidate_id": "initial_training_data",
                "selection_rank": 0,
                "fitness": initial_best,
                "round_mean_fitness": initial_mean,
                "best_observed_fitness": initial_best,
                "observed_count_before_round": len(observed),
                "note": "Initial observed training data used as first-round experimental evidence.",
            }
        )

        for round_index in range(1, rounds + 1):
            if remaining.empty:
                break

            selected_ids = strategy(observed.copy(), remaining.copy(), round_index, top_k)
            selected = evaluate_selected(remaining, selected_ids, fitness_col=evaluator_col)
            if selected.empty:
                break

            selected_fitness = selected["fitness"] if "fitness" in selected.columns else pd.Series(dtype=float)
            best_this_round = (
                float(selected_fitness.max()) if not selected_fitness.empty else None
            )
            mean_this_round = (
                float(selected_fitness.mean()) if not selected_fitness.empty else None
            )

            for _, row in selected.iterrows():
                record = row.to_dict()
                record["strategy"] = strategy_name
                record["round"] = round_index
                record["round_mean_fitness"] = mean_this_round
                record["round_best_fitness"] = best_this_round
                record["observed_count_before_round"] = len(observed)
                record["note"] = "Recommended by strategy, then evaluated by hidden virtual fitness."
                rows.append(record)

            observed = pd.concat([observed, selected], ignore_index=True, sort=False)
            remaining = remaining[~remaining["candidate_id"].isin(selected["candidate_id"])].copy()

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["best_observed_fitness"] = (
        result.groupby("strategy")["fitness"].cummax()
        if "fitness" in result.columns
        else None
    )
    return result


def summarize_recommended_positions(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    experimental = results[results["round"] > 0].copy()

    for (strategy, round_index), group in experimental.groupby(["strategy", "round"]):
        variant_count = len(group)
        position_counts = {}
        for mutations in group["mutations"]:
            for mutation in mutations if isinstance(mutations, list) else []:
                match = re.fullmatch(r"[A-Z](\d+)[A-Z]", str(mutation))
                if match:
                    position = int(match.group(1))
                    position_counts[position] = position_counts.get(position, 0) + 1

        for position, count in sorted(position_counts.items()):
            records.append(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "position": position,
                    "mutation_count": count,
                    "variant_count": variant_count,
                    "mutation_frequency": count / variant_count if variant_count else 0.0,
                }
            )

    return pd.DataFrame(records)
