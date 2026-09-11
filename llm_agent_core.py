from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pydantic import BaseModel


WT_REGION = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
PREFIX_LEN = 560
SUFFIX_LEN = 147
CONTEXT = 16
REGION_START = 561
REGION_END = 588
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class AgentPaths:
    data: Path = Path("two_vs_many.csv")
    observed: Path = Path("artifacts/observed_experiments.csv")
    checkpoint: Path = Path("artifacts/esm2_fitness_head.pt")


class CandidateSelection(BaseModel):
    candidate_id: str
    reason: str


class MutationDesignerOutput(BaseModel):
    selected: list[CandidateSelection]


class CandidateCritique(BaseModel):
    candidate_id: str
    priority: str
    recommendation_reason: str
    supporting_evidence: str
    fitness_interpretation: str
    uncertainty: str


class ScientificCriticOutput(BaseModel):
    candidate_reviews: list[CandidateCritique]
    overall_recommendation: str
    overall_limitations: str


class FitnessMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def extract_region(full_sequence: str) -> str:
    return full_sequence[PREFIX_LEN : len(full_sequence) - SUFFIX_LEN]


def get_mutation_names(region: str) -> list[str]:
    mutations = []
    for offset, (wt_aa, mut_aa) in enumerate(zip(WT_REGION, region)):
        if wt_aa != mut_aa:
            mutations.append(f"{wt_aa}{REGION_START + offset}{mut_aa}")
    return mutations


def mutation_frequency_by_position(regions) -> pd.DataFrame:
    regions = list(regions)
    variant_count = len(regions)
    rows = []

    for offset, wt_aa in enumerate(WT_REGION):
        position = REGION_START + offset
        mutation_count = sum(
            1
            for region in regions
            if len(region) > offset and region[offset] != wt_aa
        )
        rows.append(
            {
                "position": position,
                "wt": wt_aa,
                "mutation_count": mutation_count,
                "variant_count": variant_count,
                "mutation_frequency": mutation_count / variant_count if variant_count else 0.0,
            }
        )

    return pd.DataFrame(rows)


def add_mutation_features(records: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(records).copy()
    df["mutated_region"] = df["sequence"].apply(extract_region)
    df = df[df["mutated_region"].str.len() == len(WT_REGION)].copy()
    df["mutations"] = df["mutated_region"].apply(get_mutation_names)
    df["num_mutations"] = df["mutations"].apply(len)
    return df


def load_two_vs_many(path: str | Path = "two_vs_many.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    observed_df = df[df["set"] == "train"].copy()
    candidate_pool = df[df["set"] == "test"][["sequence"]].copy().reset_index(drop=True)
    candidate_pool["candidate_id"] = [f"C{i:05d}" for i in range(len(candidate_pool))]
    return observed_df, candidate_pool


def summarize_history(observed_df: pd.DataFrame, top_history_count: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_sub = add_mutation_features(observed_df)
    top_history = observed_sub.sort_values("target", ascending=False).head(top_history_count).copy()

    mutation_records = []
    for _, row in top_history.iterrows():
        for mutation in row["mutations"]:
            mutation_records.append({"mutation": mutation, "fitness": row["target"]})

    mutation_df = pd.DataFrame(mutation_records)
    if mutation_df.empty:
        mutation_summary = pd.DataFrame(columns=["mutation", "count", "mean_fitness", "max_fitness"])
    else:
        mutation_summary = (
            mutation_df.groupby("mutation")
            .agg(count=("fitness", "size"), mean_fitness=("fitness", "mean"), max_fitness=("fitness", "max"))
            .reset_index()
            .sort_values(["count", "mean_fitness"], ascending=False)
        )

    return top_history, mutation_summary


def build_analyst_context(
    observed_df: pd.DataFrame,
    top_history: pd.DataFrame,
    mutation_summary: pd.DataFrame,
) -> dict:
    return {
        "wild_type_region": WT_REGION,
        "number_of_observed_variants": len(observed_df),
        "top_variants": top_history[["mutated_region", "target"]].head(20).to_dict(orient="records"),
        "mutation_statistics": mutation_summary.head(30).to_dict(orient="records"),
    }


def build_designer_pool(
    candidate_pool: pd.DataFrame,
    mutation_summary: pd.DataFrame,
    min_count: int = 2,
    max_mutations: int = 30,
    pool_size: int = 100,
) -> pd.DataFrame:
    candidate_sub = add_mutation_features(candidate_pool)
    beneficial = (
        mutation_summary[mutation_summary["count"] >= min_count]
        .sort_values("mean_fitness", ascending=False)
        .head(max_mutations)
    )
    beneficial_score = dict(zip(beneficial["mutation"], beneficial["mean_fitness"]))
    candidate_sub["historical_prior"] = candidate_sub["mutations"].apply(
        lambda mutations: sum(beneficial_score.get(mutation, 0.0) for mutation in mutations)
    )
    return candidate_sub.sort_values("historical_prior", ascending=False).head(pool_size).copy()


def candidate_records(designer_pool: pd.DataFrame) -> list[dict]:
    return designer_pool[["candidate_id", "mutated_region", "mutations", "num_mutations"]].to_dict(
        orient="records"
    )


def compact_candidate_records(
    designer_pool: pd.DataFrame,
    limit: int = 40,
    max_mutations: int | None = None,
) -> list[dict]:
    filtered = designer_pool
    if max_mutations is not None and "num_mutations" in designer_pool.columns:
        filtered = designer_pool[designer_pool["num_mutations"] <= max_mutations]

    available_columns = [
        column
        for column in ["candidate_id", "mutations", "num_mutations"]
        if column in filtered.columns
    ]
    return filtered.head(limit)[available_columns].to_dict(orient="records")


def rule_based_designer_result(candidates: list[dict], n: int = 30) -> MutationDesignerOutput:
    selected = []
    for candidate in candidates[:n]:
        mutations = candidate.get("mutations", [])
        selected.append(
            CandidateSelection(
                candidate_id=str(candidate["candidate_id"]),
                reason=(
                    "Selected by local fallback ranking from the pre-filtered designer pool "
                    f"after the LLM API call was unavailable; contains {len(mutations)} mutations."
                ),
            )
        )
    return MutationDesignerOutput(selected=selected)


def run_data_analyst(client, model: str, context: dict) -> str:
    prompt = f"""
You are the Data Analyst module of an AAV directed-evolution benchmark agent.

Wild-type mutation region:
{context["wild_type_region"]}

Number of observed variants:
{context["number_of_observed_variants"]}

Top historical variants:
{json.dumps(context["top_variants"], indent=2)}

Mutation statistics:
{json.dumps(context["mutation_statistics"], indent=2)}

Please provide a concise scientific analysis. Separate observed evidence from hypotheses,
do not claim causality from correlation, and do not invent experimental evidence.
"""
    return client.responses.create(model=model, input=prompt).output_text


def generate_hypotheses(client, model: str, analyst_report: str) -> str:
    prompt = f"""
You are the Hypothesis Generator module of an AAV directed-evolution benchmark agent.

Below is the historical Data Analyst report:

{analyst_report}

Generate 5 testable hypotheses for candidate prioritization. Include target positions,
substitution patterns, historical evidence, scientific rationale, and uncertainty.
Do not invent experimental evidence.
"""
    return client.responses.create(model=model, input=prompt).output_text


def mutation_designer(
    client,
    model: str,
    analyst_report: str,
    hypotheses: str,
    candidates: list[dict],
    n: int = 30,
    max_candidates_for_llm: int = 40,
    fallback_on_error: bool = True,
):
    compact_candidates = candidates[:max_candidates_for_llm]
    prompt = f"""
You are the Mutation Designer module of an AAV directed-evolution benchmark agent.

DATA ANALYST REPORT
{analyst_report[:2500]}

SCIENTIFIC HYPOTHESES
{hypotheses[:2000]}

AVAILABLE CANDIDATES
{json.dumps(compact_candidates, separators=(",", ":"))}

Select exactly {n} candidates for the next computational fitness-screening step.
Only select candidate IDs from the supplied list, prefer historically supported and
hypothesis-relevant candidates, maintain diversity, and do not invent fitness.
"""
    try:
        return client.responses.parse(model=model, input=prompt, text_format=MutationDesignerOutput).output_parsed
    except Exception:
        if not fallback_on_error:
            raise
        return rule_based_designer_result(compact_candidates, n=n)


def merge_designed_candidates(designer_pool: pd.DataFrame, designer_result) -> pd.DataFrame:
    selection_df = pd.DataFrame([item.model_dump() for item in designer_result.selected])
    selection_df = selection_df.drop_duplicates(subset="candidate_id").head(30).copy()
    return designer_pool.merge(selection_df, on="candidate_id", how="inner").copy()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_region_and_context(sequence: str, context: int = CONTEXT) -> tuple[str, int, int]:
    region = extract_region(sequence)
    left_start = max(0, PREFIX_LEN - context)
    left = sequence[left_start:PREFIX_LEN]
    suffix_start = len(sequence) - SUFFIX_LEN
    right = sequence[suffix_start : min(len(sequence), suffix_start + context)]
    context_sequence = left + region + right
    region_start = len(left)
    return context_sequence, region_start, region_start + len(region)


def load_fitness_model(checkpoint_path: str | Path, device: torch.device) -> FitnessMLP:
    model = FitnessMLP()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def get_esm_embeddings(sequences, esm_model, batch_converter, device: torch.device, batch_size: int = 16) -> np.ndarray:
    sequences = list(sequences)
    all_embeddings = []

    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            batch_info = []
            for index, sequence in enumerate(batch_sequences):
                context_sequence, region_start, region_end = extract_region_and_context(sequence)
                batch_info.append((f"seq_{start + index}", context_sequence, region_start, region_end))

            _, _, tokens = batch_converter([(name, seq) for name, seq, _, _ in batch_info])
            output = esm_model(tokens.to(device), repr_layers=[12], return_contacts=False)
            representations = output["representations"][12]

            for index, (_, _, region_start, region_end) in enumerate(batch_info):
                region_embedding = representations[index, region_start + 1 : region_end + 1, :]
                all_embeddings.append(region_embedding.mean(dim=0).cpu().numpy())

    return np.stack(all_embeddings).astype(np.float32)


def evaluate_fitness(candidate_df: pd.DataFrame, esm_model, batch_converter, fitness_model, device, batch_size=16):
    embeddings = get_esm_embeddings(candidate_df["sequence"], esm_model, batch_converter, device, batch_size)
    x_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = fitness_model(x_tensor).cpu().numpy()
    result = candidate_df.copy()
    result["predicted_fitness"] = predictions
    return result


def rule_based_scientific_critic(top_candidates: list[dict]) -> ScientificCriticOutput:
    scores = [float(candidate.get("predicted_fitness", 0.0)) for candidate in top_candidates]
    if not scores:
        return ScientificCriticOutput(
            candidate_reviews=[],
            overall_recommendation="No candidates were supplied for review.",
            overall_limitations="Local fallback critic was used and no candidates were available.",
        )

    ordered_scores = sorted(scores, reverse=True)
    high_cutoff = ordered_scores[min(2, len(ordered_scores) - 1)]
    medium_cutoff = ordered_scores[min(5, len(ordered_scores) - 1)]
    reviews = []

    for candidate in top_candidates:
        predicted = float(candidate.get("predicted_fitness", 0.0))
        mutations = candidate.get("mutations", [])
        if predicted >= high_cutoff:
            priority = "High"
        elif predicted >= medium_cutoff:
            priority = "Medium"
        else:
            priority = "Low"

        reviews.append(
            CandidateCritique(
                candidate_id=str(candidate["candidate_id"]),
                priority=priority,
                recommendation_reason=(
                    f"Ranked by the trained fitness predictor with predicted fitness {predicted:.3f}; "
                    f"contains {len(mutations)} mutations."
                ),
                supporting_evidence=(
                    "Candidate came from the pre-filtered designer pool enriched for mutations observed "
                    "in high-fitness historical variants."
                ),
                fitness_interpretation=(
                    "The value is a model prediction for prioritization, not an experimental measurement."
                ),
                uncertainty=(
                    "Local fallback critic used because the LLM API was unavailable or rate-limited; "
                    "epistasis and distribution shift remain possible."
                ),
            )
        )

    return ScientificCriticOutput(
        candidate_reviews=reviews,
        overall_recommendation=(
            "Prioritize candidates with High priority first, then Medium priority candidates as follow-up "
            "comparisons for the same mutation patterns."
        ),
        overall_limitations=(
            "Local fallback Scientific Critic was used. Explanations are rule-based and conservative, "
            "not LLM-generated scientific reasoning."
        ),
    )


def scientific_critic(
    client,
    model: str,
    top_candidates: list[dict],
    analyst_report: str,
    hypotheses: str,
    fallback_on_error: bool = True,
):
    prompt = f"""
You are the Scientific Critic module of an AAV directed-evolution computational benchmark agent.

HISTORICAL DATA ANALYSIS
{analyst_report[:2000]}

SCIENTIFIC HYPOTHESES
{hypotheses[:1500]}

TOP CANDIDATES
{json.dumps(top_candidates, separators=(",", ":"))}

Evaluate all supplied candidates. Explain priority, supporting evidence, fitness interpretation,
and uncertainty. Predicted fitness is not experimental validation. Do not invent results.
"""
    try:
        return client.responses.parse(model=model, input=prompt, text_format=ScientificCriticOutput).output_parsed
    except Exception:
        if not fallback_on_error:
            raise
        return rule_based_scientific_critic(top_candidates)


def knowledge_enhanced_scientific_critic(
    client,
    model: str,
    top_candidates: list[dict],
    knowledge_context: str,
    graph_triples: list[tuple[str, str, str]] | pd.DataFrame,
    fallback_on_error: bool = True,
):
    if isinstance(graph_triples, pd.DataFrame):
        graph_records = graph_triples.head(120).to_dict(orient="records")
    else:
        graph_records = [
            {"subject": subject, "predicate": predicate, "object": obj}
            for subject, predicate, obj in list(graph_triples)[:120]
        ]

    prompt = f"""
You are the Knowledge-Enhanced Scientific Critic module of an AAV directed-evolution benchmark agent.

Use the supplied knowledge base to re-rank and explain candidate recommendations. The candidates
already include model-predicted fitness and, for this offline benchmark only, true_fitness from the
held-out table. Treat true_fitness as retrospective evaluation evidence, not information available
to a real wet-lab design loop.

KNOWLEDGE RULES AND PROPERTY CONTEXT
{knowledge_context[:3500]}

KNOWLEDGE GRAPH TRIPLES
{json.dumps(graph_records, separators=(",", ":"))}

CANDIDATES TO REVIEW
{json.dumps(top_candidates, separators=(",", ":"))}

Evaluate every supplied candidate. Discuss mutation-count risk, physicochemical/conservative
substitution support, knowledge-graph evidence, predicted-vs-true fitness agreement, and uncertainty.
Do not invent new measurements or candidate IDs.
"""
    try:
        return client.responses.parse(model=model, input=prompt, text_format=ScientificCriticOutput).output_parsed
    except Exception:
        if not fallback_on_error:
            raise
        return rule_based_scientific_critic(top_candidates)


def build_critic_candidates(top_k: pd.DataFrame) -> list[dict]:
    return [
        {
            "rank": int(row["rank"]),
            "candidate_id": str(row["candidate_id"]),
            "mutations": list(row["mutations"]),
            "predicted_fitness": float(row["predicted_fitness"]),
            "mutation_designer_reason": str(row["reason"]),
        }
        for _, row in top_k.iterrows()
    ]


def merge_final_recommendations(top_k: pd.DataFrame, critic_result) -> pd.DataFrame:
    critic_df = pd.DataFrame([item.model_dump() for item in critic_result.candidate_reviews])
    return top_k.merge(critic_df, on="candidate_id", how="left").copy()


def attach_candidate_truth(candidates: pd.DataFrame, test_truth: pd.DataFrame) -> pd.DataFrame:
    clean_candidates = candidates.drop(
        columns=[column for column in ["true_fitness", "prediction_error"] if column in candidates.columns]
    )
    truth = test_truth[["target"]].copy().reset_index(drop=True)
    truth["candidate_id"] = [f"C{i:05d}" for i in range(len(truth))]
    truth = truth.rename(columns={"target": "true_fitness"})

    result = clean_candidates.merge(truth[["candidate_id", "true_fitness"]], on="candidate_id", how="left")
    if "predicted_fitness" in result.columns:
        result["prediction_error"] = result["predicted_fitness"] - result["true_fitness"]
    return result
