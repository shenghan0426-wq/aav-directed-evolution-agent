from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .knowledge_base import (
        apply_mutation_count_constraint,
        score_mutation_rules,
        weighted_knowledge_score,
    )
    from .llm_agent_core import WT_REGION, build_designer_pool, summarize_history
except ImportError:
    from knowledge_base import (
        apply_mutation_count_constraint,
        score_mutation_rules,
        weighted_knowledge_score,
    )
    from llm_agent_core import WT_REGION, build_designer_pool, summarize_history


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_DIR / "artifacts"


def _with_candidate_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    if "candidate_id" not in result.columns:
        result["candidate_id"] = [f"C{i:05d}" for i in range(len(result))]
    return result


def _load_predictions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    predictions = pd.read_csv(path).reset_index(drop=True)
    predictions["candidate_id"] = [f"C{i:05d}" for i in range(len(predictions))]
    return predictions


def recommend_mutations_for_region(
    wild_type_region: str,
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    top_k: int = 10,
    max_mutations: int = 4,
) -> pd.DataFrame:
    if wild_type_region != WT_REGION:
        raise ValueError(
            "This demo is calibrated for the AAV mutation region "
            f"{WT_REGION!r}. Please use the default region for reproducible scoring."
        )

    candidate_df = _with_candidate_ids(candidate_df)
    if predictions is not None:
        predictions = _with_candidate_ids(predictions)

    _, mutation_summary = summarize_history(observed_df)
    designer_pool = build_designer_pool(
        candidate_df[["sequence", "candidate_id"]],
        mutation_summary,
        pool_size=len(candidate_df),
    )
    constrained = apply_mutation_count_constraint(designer_pool, max_mutations=max_mutations)
    if constrained.empty:
        minimum_available = int(designer_pool["num_mutations"].min())
        raise ValueError(
            "No candidates satisfy the current maximum mutation constraint. "
            f"The smallest candidate in this pool has {minimum_available} mutations; "
            f"please set Maximum mutations to at least {minimum_available}."
        )

    rule_scores = constrained["mutations"].apply(score_mutation_rules).apply(pd.Series)
    scored = pd.concat([constrained.reset_index(drop=True), rule_scores.reset_index(drop=True)], axis=1)
    scored = weighted_knowledge_score(scored, historical_weight=0.7, rule_weight=0.3)

    if predictions is not None and "predicted_fitness" in predictions.columns:
        scored = scored.merge(
            predictions[["candidate_id", "predicted_fitness"]],
            on="candidate_id",
            how="left",
        )
    else:
        scored["predicted_fitness"] = scored["historical_prior"]

    if "target" in candidate_df.columns:
        scored = scored.merge(candidate_df[["candidate_id", "target"]], on="candidate_id", how="left")
        scored = scored.rename(columns={"target": "true_fitness"})

    scored["final_demo_score"] = (
        0.6 * scored["knowledge_enhanced_score"]
        + 0.4 * scored["predicted_fitness"].rank(pct=True).fillna(0.0)
    )
    scored["recommendation_reason"] = scored.apply(
        lambda row: (
            f"mutation_count={int(row['num_mutations'])}; "
            f"historical_prior={row['historical_prior']:.3f}; "
            f"rule_score={row['rule_score']:.3f}; "
            "prioritizes historically supported low-complexity AAV variants."
        ),
        axis=1,
    )

    columns = [
        "candidate_id",
        "mutated_region",
        "mutations",
        "num_mutations",
        "predicted_fitness",
        "knowledge_enhanced_score",
        "final_demo_score",
        "recommendation_reason",
    ]
    if "true_fitness" in scored.columns:
        columns.insert(5, "true_fitness")

    return (
        scored.sort_values("final_demo_score", ascending=False)
        .head(top_k)[columns]
        .reset_index(drop=True)
    )


def load_demo_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train = pd.read_csv(PROJECT_DIR / "train.csv")
    test = pd.read_csv(PROJECT_DIR / "test.csv")
    predictions = _load_predictions(ARTIFACTS / "test_esm2_mlp_predictions.csv")
    return train, test, predictions


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="AAV Directed Evolution Demo", layout="wide")
    st.title("AAV Directed Evolution Agent Demo")

    wild_type_region = st.text_area(
        "Wild-type mutation region",
        value=WT_REGION,
        height=80,
    ).strip()
    col_a, col_b = st.columns(2)
    with col_a:
        top_k = st.slider("Top-k recommendations", min_value=3, max_value=20, value=10)
    with col_b:
        max_mutations = st.slider("Maximum mutations", min_value=1, max_value=12, value=4)

    train, test, predictions = load_demo_inputs()

    if st.button("Recommend mutations", type="primary"):
        try:
            recommendations = recommend_mutations_for_region(
                wild_type_region,
                train,
                test,
                predictions=predictions,
                top_k=top_k,
                max_mutations=max_mutations,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.dataframe(recommendations, use_container_width=True, hide_index=True)
        st.download_button(
            "Download recommendations",
            data=recommendations.to_csv(index=False),
            file_name="demo_recommendations.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    run_streamlit_app()
