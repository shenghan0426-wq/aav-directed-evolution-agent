from __future__ import annotations

import argparse
import os
from pathlib import Path

import esm
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from llm_agent_core import (
    AgentPaths,
    build_analyst_context,
    build_critic_candidates,
    build_designer_pool,
    candidate_records,
    choose_device,
    evaluate_fitness,
    generate_hypotheses,
    load_fitness_model,
    load_two_vs_many,
    merge_designed_candidates,
    merge_final_recommendations,
    mutation_designer,
    run_data_analyst,
    scientific_critic,
    summarize_history,
)


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AAV LLM agent workflow.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--designer-pool-size", type=int, default=100)
    parser.add_argument("--designer-selection-size", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build historical summaries and candidate pool; skip OpenAI and ESM inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    paths = AgentPaths(
        data=BASE_DIR / AgentPaths.data,
        observed=BASE_DIR / AgentPaths.observed,
        checkpoint=BASE_DIR / AgentPaths.checkpoint,
    )

    observed_df, candidate_pool = load_two_vs_many(paths.data)
    top_history, mutation_summary = summarize_history(observed_df)
    analyst_context = build_analyst_context(observed_df, top_history, mutation_summary)
    designer_pool = build_designer_pool(
        candidate_pool,
        mutation_summary,
        pool_size=args.designer_pool_size,
    )
    records = candidate_records(designer_pool)

    print("Observed experiments:", len(observed_df))
    print("Designer candidate pool:", len(designer_pool))

    if args.prepare_only:
        print(pd.DataFrame(records).head(10))
        return

    client = OpenAI()
    analyst_report = run_data_analyst(client, args.model, analyst_context)
    hypotheses = generate_hypotheses(client, args.model, analyst_report)
    designer_result = mutation_designer(
        client,
        args.model,
        analyst_report,
        hypotheses,
        records,
        n=args.designer_selection_size,
    )
    designed_candidates = merge_designed_candidates(designer_pool, designer_result)

    device = choose_device()
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.to(device).eval()
    for parameter in esm_model.parameters():
        parameter.requires_grad = False

    fitness_model = load_fitness_model(paths.checkpoint, device)
    scored_candidates = evaluate_fitness(
        designed_candidates,
        esm_model,
        alphabet.get_batch_converter(),
        fitness_model,
        device,
        batch_size=args.batch_size,
    )

    top_k = (
        scored_candidates.sort_values("predicted_fitness", ascending=False)
        .head(args.top_k)
        .copy()
        .reset_index(drop=True)
    )
    top_k.insert(0, "rank", range(1, len(top_k) + 1))

    critic_result = scientific_critic(
        client,
        args.model,
        build_critic_candidates(top_k),
        analyst_report,
        hypotheses,
    )
    final_recommendations = merge_final_recommendations(top_k, critic_result)

    output_path = BASE_DIR / "artifacts/llm_agent_final_recommendations.csv"
    final_recommendations.to_csv(output_path, index=False)

    print(final_recommendations[["rank", "candidate_id", "predicted_fitness", "priority"]])
    print("\n========== OVERALL RECOMMENDATION ==========")
    print(critic_result.overall_recommendation)
    print("\n========== MAJOR LIMITATIONS ==========")
    print(critic_result.overall_limitations)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
