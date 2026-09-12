from __future__ import annotations

import random
import re

import pandas as pd
from pydantic import BaseModel


class RoundCandidateDecision(BaseModel):
    candidate_id: str
    recommendation_reason: str


class RoundCriticDecision(BaseModel):
    selected: list[RoundCandidateDecision]
    critic_feedback: str
    successful_patterns: list[str]
    failed_patterns: list[str]
    next_round_strategy: str
    limitations: str


def mutation_positions(mutations) -> set[int]:
    positions = set()
    for mutation in mutations if isinstance(mutations, list) else []:
        match = re.fullmatch(r"[A-Z](\d+)[A-Z]", str(mutation))
        if match:
            positions.add(int(match.group(1)))
    return positions


def _safe_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if series.nunique(dropna=True) <= 1:
        return pd.Series([0.5] * len(series), index=series.index)
    return series.rank(pct=True).fillna(0.0)


def summarize_feedback(
    observed_df: pd.DataFrame,
    fitness_col: str = "target",
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
) -> dict:
    if observed_df.empty or fitness_col not in observed_df.columns or "mutations" not in observed_df.columns:
        return {
            "successful_positions": [],
            "failed_positions": [],
            "successful_mutations": [],
            "failed_mutations": [],
            "fitness_threshold_high": None,
            "fitness_threshold_low": None,
        }

    observed = observed_df.dropna(subset=[fitness_col]).copy()
    if observed.empty:
        return {
            "successful_positions": [],
            "failed_positions": [],
            "successful_mutations": [],
            "failed_mutations": [],
            "fitness_threshold_high": None,
            "fitness_threshold_low": None,
        }
    if "virtual_feedback_round" in observed.columns:
        feedback_rounds = observed["virtual_feedback_round"].dropna()
        feedback_rounds = feedback_rounds[feedback_rounds > 0]
        if not feedback_rounds.empty:
            observed = observed[observed["virtual_feedback_round"] == feedback_rounds.max()].copy()

    high_threshold = float(observed[fitness_col].quantile(top_quantile))
    low_threshold = float(observed[fitness_col].quantile(bottom_quantile))
    successful = observed[observed[fitness_col] >= high_threshold]
    failed = observed[observed[fitness_col] <= low_threshold]

    successful_mutations = sorted(
        {str(mutation) for mutations in successful["mutations"] if isinstance(mutations, list) for mutation in mutations}
    )
    failed_mutations = sorted(
        {str(mutation) for mutations in failed["mutations"] if isinstance(mutations, list) for mutation in mutations}
    )
    successful_positions = sorted({position for mutations in successful["mutations"] for position in mutation_positions(mutations)})
    failed_positions = sorted({position for mutations in failed["mutations"] for position in mutation_positions(mutations)})

    return {
        "successful_positions": successful_positions,
        "failed_positions": failed_positions,
        "successful_mutations": successful_mutations,
        "failed_mutations": failed_mutations,
        "fitness_threshold_high": high_threshold,
        "fitness_threshold_low": low_threshold,
    }


def score_candidates_with_feedback(
    candidate_df: pd.DataFrame,
    feedback: dict,
    max_mutations: int = 4,
    predicted_col: str = "predicted_fitness",
    knowledge_col: str = "knowledge_enhanced_score",
) -> pd.DataFrame:
    scored = candidate_df.copy()
    if scored.empty:
        return scored

    successful_positions = set(feedback.get("successful_positions", []))
    failed_positions = set(feedback.get("failed_positions", []))

    scored["feedback_supported_positions"] = scored["mutations"].apply(
        lambda mutations: len(mutation_positions(mutations) & successful_positions)
    )
    scored["feedback_failed_positions"] = scored["mutations"].apply(
        lambda mutations: len(mutation_positions(mutations) & failed_positions)
    )
    scored["mutation_count_penalty"] = scored.get("num_mutations", pd.Series(0, index=scored.index)).apply(
        lambda count: max(0, int(count) - max_mutations)
    )
    scored["feedback_score"] = (
        scored["feedback_supported_positions"]
        - 1.5 * scored["feedback_failed_positions"]
        - 0.5 * scored["mutation_count_penalty"]
    )

    predicted_rank = _safe_rank(scored[predicted_col]) if predicted_col in scored.columns else pd.Series(0.0, index=scored.index)
    knowledge_rank = _safe_rank(scored[knowledge_col]) if knowledge_col in scored.columns else pd.Series(0.0, index=scored.index)
    feedback_rank = _safe_rank(scored["feedback_score"])
    scored["feedback_agent_score"] = 0.60 * feedback_rank + 0.25 * predicted_rank + 0.15 * knowledge_rank
    scored["critic_feedback"] = scored.apply(
        lambda row: (
            f"supported_positions={int(row['feedback_supported_positions'])}; "
            f"failed_positions={int(row['feedback_failed_positions'])}; "
            f"mutation_count_penalty={int(row['mutation_count_penalty'])}"
        ),
        axis=1,
    )
    return scored


def build_feedback_driven_agent_strategy(
    max_mutations: int = 4,
    predicted_col: str = "predicted_fitness",
    knowledge_col: str = "knowledge_enhanced_score",
    fitness_col: str = "target",
):
    def strategy(observed_df: pd.DataFrame, candidate_df: pd.DataFrame, round_index: int, top_k: int) -> list[str]:
        feedback = summarize_feedback(observed_df, fitness_col=fitness_col)
        scored = score_candidates_with_feedback(
            candidate_df,
            feedback,
            max_mutations=max_mutations,
            predicted_col=predicted_col,
            knowledge_col=knowledge_col,
        )
        constrained = scored[scored.get("num_mutations", 0) <= max_mutations].copy()
        if constrained.empty:
            constrained = scored
        return (
            constrained.sort_values(["feedback_agent_score", predicted_col], ascending=False)
            .head(top_k)["candidate_id"]
            .astype(str)
            .tolist()
        )

    return strategy


def build_round_candidate_records(candidate_df: pd.DataFrame, limit: int = 80) -> list[dict]:
    columns = [
        "candidate_id",
        "mutations",
        "num_mutations",
        "predicted_fitness",
        "knowledge_enhanced_score",
        "feedback_score",
        "feedback_agent_score",
        "critic_feedback",
    ]
    available_columns = [column for column in columns if column in candidate_df.columns]
    records = []
    for _, row in candidate_df.head(limit)[available_columns].iterrows():
        record = row.to_dict()
        for key in ["predicted_fitness", "knowledge_enhanced_score", "feedback_score", "feedback_agent_score"]:
            if key in record and pd.notna(record[key]):
                record[key] = round(float(record[key]), 4)
        if "num_mutations" in record and pd.notna(record["num_mutations"]):
            record["num_mutations"] = int(record["num_mutations"])
        if "mutations" in record and not isinstance(record["mutations"], list):
            record["mutations"] = []
        records.append(record)
    return records


def build_previous_round_records(observed_df: pd.DataFrame, fitness_col: str = "target", limit: int = 20) -> list[dict]:
    if "virtual_feedback_round" not in observed_df.columns:
        return []
    feedback_rounds = observed_df["virtual_feedback_round"].dropna()
    feedback_rounds = feedback_rounds[feedback_rounds > 0]
    if feedback_rounds.empty:
        return []

    previous = observed_df[observed_df["virtual_feedback_round"] == feedback_rounds.max()].copy()
    if fitness_col not in previous.columns:
        return []
    previous = previous.sort_values(fitness_col, ascending=False).head(limit)
    records = []
    for _, row in previous.iterrows():
        records.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "mutations": row.get("mutations", []),
                "num_mutations": int(row.get("num_mutations", 0)),
                "true_fitness": round(float(row[fitness_col]), 4),
            }
        )
    return records


def rule_based_round_critic_decision(candidate_records: list[dict], top_k: int = 10) -> RoundCriticDecision:
    selected = [
        RoundCandidateDecision(
            candidate_id=str(record["candidate_id"]),
            recommendation_reason=(
                "Selected by fallback feedback-aware ranking because the OpenAI per-round call "
                "was unavailable; prioritizes feedback-supported positions, model score, and low mutation count."
            ),
        )
        for record in candidate_records[:top_k]
    ]
    return RoundCriticDecision(
        selected=selected,
        critic_feedback=(
            "Fallback critic used structured previous-round feedback to bias candidate selection. "
            "No natural-language OpenAI critique was generated."
        ),
        successful_patterns=[],
        failed_patterns=[],
        next_round_strategy=(
            "Continue prioritizing candidates with feedback-supported positions, moderate predicted fitness, "
            "and no more than the configured maximum mutation count."
        ),
        limitations="Fallback decision; no live OpenAI reasoning was used for this round.",
    )


def openai_round_critic_decision(
    client,
    model: str,
    round_index: int,
    previous_round_records: list[dict],
    feedback: dict,
    candidate_records: list[dict],
    top_k: int = 10,
    fallback_on_error: bool = True,
) -> RoundCriticDecision:
    prompt = f"""
You are the per-round Scientific Critic and Mutation Designer for an AAV directed-evolution agent.

This is virtual evolution round {round_index}. You must use previous experimental feedback to choose
the next Top-{top_k} candidates from the supplied candidate list only.

PREVIOUS ROUND TRUE-FITNESS RESULTS
{previous_round_records}

STRUCTURED FEEDBACK FROM OBSERVED RESULTS
{feedback}

AVAILABLE CANDIDATES
{candidate_records}

Instructions:
- Select exactly {top_k} candidate IDs from AVAILABLE CANDIDATES.
- Do not invent candidate IDs or fitness values.
- Prefer candidates that preserve successful positions or mutations from previous feedback.
- Avoid failed positions or high-risk high-mutation backgrounds unless there is a clear reason.
- Explain what the Critic learned from the previous round and how it changes this round's recommendation.
- Treat predicted_fitness as model output and true_fitness from previous rounds as experimental feedback.
"""
    try:
        return client.responses.parse(model=model, input=prompt, text_format=RoundCriticDecision).output_parsed
    except Exception:
        if not fallback_on_error:
            raise
        return rule_based_round_critic_decision(candidate_records, top_k=top_k)


def select_openai_feedback_candidates(
    client,
    model: str,
    observed_df: pd.DataFrame,
    available_candidates: pd.DataFrame,
    round_index: int,
    top_k: int = 10,
    candidate_pool_size: int = 80,
    max_mutations: int = 4,
    predicted_col: str = "predicted_fitness",
    knowledge_col: str = "knowledge_enhanced_score",
    fitness_col: str = "target",
) -> tuple[list[str], RoundCriticDecision, pd.DataFrame]:
    feedback = summarize_feedback(observed_df, fitness_col=fitness_col)
    scored = score_candidates_with_feedback(
        available_candidates,
        feedback,
        max_mutations=max_mutations,
        predicted_col=predicted_col,
        knowledge_col=knowledge_col,
    )
    constrained = scored[scored.get("num_mutations", 0) <= max_mutations].copy()
    if constrained.empty:
        constrained = scored
    ranked = constrained.sort_values(["feedback_agent_score", predicted_col], ascending=False).head(candidate_pool_size)
    candidate_records = build_round_candidate_records(ranked, limit=candidate_pool_size)
    previous_round_records = build_previous_round_records(observed_df, fitness_col=fitness_col)
    decision = openai_round_critic_decision(
        client,
        model,
        round_index,
        previous_round_records,
        feedback,
        candidate_records,
        top_k=top_k,
    )

    valid_ids = set(ranked["candidate_id"].astype(str))
    selected_ids = []
    for item in decision.selected:
        candidate_id = str(item.candidate_id)
        if candidate_id in valid_ids and candidate_id not in selected_ids:
            selected_ids.append(candidate_id)
        if len(selected_ids) >= top_k:
            break
    if len(selected_ids) < top_k:
        for candidate_id in ranked["candidate_id"].astype(str):
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
            if len(selected_ids) >= top_k:
                break

    return selected_ids, decision, ranked


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

            selected_for_history = selected.copy()
            selected_for_history["virtual_feedback_round"] = round_index
            observed = pd.concat([observed, selected_for_history], ignore_index=True, sort=False)
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
