import unittest

import pandas as pd

from aav_baseline.knowledge_base import (
    amino_acid_properties,
    build_knowledge_context,
    build_mutation_knowledge_graph,
    apply_mutation_count_constraint,
    min_max_normalize,
    score_mutation_rules,
    weighted_knowledge_score,
)
from aav_baseline.llm_agent_core import (
    REGION_START,
    WT_REGION,
    add_mutation_features,
    attach_candidate_truth,
    compact_candidate_records,
    extract_region,
    get_mutation_names,
    knowledge_enhanced_scientific_critic,
    mutation_frequency_by_position,
    rule_based_designer_result,
    rule_based_scientific_critic,
)
from aav_baseline.model_evaluation import evaluate_predictions, one_hot_encode_regions, predict_with_fitness_head
from aav_baseline.report_figures import summarize_rounds
from aav_baseline.virtual_evolution import (
    RoundCandidateDecision,
    RoundCriticDecision,
    build_feedback_driven_agent_strategy,
    select_model_top_candidates,
    select_openai_feedback_candidates,
    select_random_candidates,
    simulate_iterative_evolution,
    summarize_feedback,
    summarize_recommended_positions,
)
from aav_baseline.app_demo import recommend_mutations_for_region


class LlmAgentCoreTest(unittest.TestCase):
    def test_extract_region_uses_configured_prefix_and_suffix(self):
        sequence = "P" * 560 + WT_REGION + "S" * 147

        self.assertEqual(extract_region(sequence), WT_REGION)

    def test_get_mutation_names_reports_one_based_absolute_positions(self):
        mutated = "A" + WT_REGION[1:10] + "A" + WT_REGION[11:]

        self.assertEqual(
            get_mutation_names(mutated),
            [f"{WT_REGION[0]}{REGION_START}A", f"{WT_REGION[10]}{REGION_START + 10}A"],
        )

    def test_add_mutation_features_keeps_substitution_only_rows(self):
        rows = [
            {"candidate_id": "C00001", "sequence": "P" * 560 + WT_REGION + "S" * 147},
            {"candidate_id": "C00002", "sequence": "P" * 560 + WT_REGION[:-1] + "S" * 147},
        ]

        result = add_mutation_features(rows)

        self.assertEqual(list(result["candidate_id"]), ["C00001"])
        self.assertEqual(result.iloc[0]["mutations"], [])
        self.assertEqual(result.iloc[0]["num_mutations"], 0)

    def test_mutation_frequency_by_position_counts_mutated_sites(self):
        regions = [
            "A" + WT_REGION[1:],
            "A" + WT_REGION[1:10] + "A" + WT_REGION[11:],
        ]

        result = mutation_frequency_by_position(regions)

        self.assertEqual(result.loc[result["position"] == REGION_START, "mutation_count"].iloc[0], 2)
        self.assertEqual(result.loc[result["position"] == REGION_START + 10, "mutation_count"].iloc[0], 1)
        self.assertEqual(result["variant_count"].iloc[0], 2)

    def test_compact_candidate_records_limits_fields_and_rows(self):
        df = pd.DataFrame(
            [
                {
                    "candidate_id": "C00001",
                    "mutated_region": WT_REGION,
                    "mutations": ["D561A"],
                    "num_mutations": 1,
                    "historical_prior": 3.5,
                    "sequence": "x",
                }
            ]
        )

        result = compact_candidate_records(df, limit=1)

        self.assertEqual(result, [{"candidate_id": "C00001", "mutations": ["D561A"], "num_mutations": 1}])

    def test_compact_candidate_records_filters_by_max_mutations(self):
        df = pd.DataFrame(
            [
                {"candidate_id": "C00001", "mutations": ["D561A"] * 4, "num_mutations": 4},
                {"candidate_id": "C00002", "mutations": ["D561A"] * 5, "num_mutations": 5},
            ]
        )

        result = compact_candidate_records(df, limit=10, max_mutations=4)

        self.assertEqual([item["candidate_id"] for item in result], ["C00001"])

    def test_rule_based_designer_result_returns_structured_selection(self):
        candidates = [
            {"candidate_id": "C00001", "mutations": ["D561A"], "num_mutations": 1},
            {"candidate_id": "C00002", "mutations": ["E562A"], "num_mutations": 1},
        ]

        result = rule_based_designer_result(candidates, n=1)

        self.assertEqual(len(result.selected), 1)
        self.assertEqual(result.selected[0].candidate_id, "C00001")

    def test_rule_based_scientific_critic_reviews_all_candidates(self):
        candidates = [
            {"candidate_id": "C00001", "mutations": ["D561A"], "predicted_fitness": 3.0},
            {"candidate_id": "C00002", "mutations": ["E562A"], "predicted_fitness": 1.0},
        ]

        result = rule_based_scientific_critic(candidates)

        self.assertEqual(len(result.candidate_reviews), 2)
        self.assertEqual(result.candidate_reviews[0].priority, "High")
        self.assertIn("fallback", result.overall_limitations.lower())

    def test_knowledge_enhanced_scientific_critic_falls_back_to_structured_reviews(self):
        class FailingResponses:
            def parse(self, **kwargs):
                raise RuntimeError("rate limited")

        class FailingClient:
            responses = FailingResponses()

        candidates = [
            {
                "candidate_id": "C00001",
                "mutations": ["D561A"],
                "predicted_fitness": 3.0,
                "true_fitness": 2.0,
                "knowledge_enhanced_score": 0.9,
            },
        ]

        result = knowledge_enhanced_scientific_critic(
            FailingClient(),
            "test-model",
            candidates,
            knowledge_context="D561A: conservative substitution",
            graph_triples=[("C00001", "contains", "D561A")],
        )

        self.assertEqual(len(result.candidate_reviews), 1)
        self.assertEqual(result.candidate_reviews[0].candidate_id, "C00001")
        self.assertIn("fallback", result.overall_limitations.lower())

    def test_attach_candidate_truth_adds_target_without_changing_rows(self):
        candidates = pd.DataFrame(
            [
                {"candidate_id": "C00001", "predicted_fitness": 1.5},
                {"candidate_id": "C00000", "predicted_fitness": 2.5},
            ]
        )
        test_truth = pd.DataFrame(
            [
                {"target": -1.0},
                {"target": 4.0},
            ]
        )

        result = attach_candidate_truth(candidates, test_truth)

        self.assertEqual(list(result["candidate_id"]), ["C00001", "C00000"])
        self.assertEqual(list(result["true_fitness"]), [4.0, -1.0])

    def test_attach_candidate_truth_replaces_existing_truth_columns(self):
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "C00000",
                    "predicted_fitness": 2.5,
                    "true_fitness": 99.0,
                    "prediction_error": -99.0,
                },
            ]
        )
        test_truth = pd.DataFrame([{"target": 4.0}])

        result = attach_candidate_truth(candidates, test_truth)

        self.assertEqual(result.iloc[0]["true_fitness"], 4.0)
        self.assertEqual(result.iloc[0]["prediction_error"], -1.5)

    def test_evaluate_predictions_returns_core_metrics(self):
        result = evaluate_predictions([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], top_k=1)

        self.assertGreaterEqual(set(result), {"mse", "pearson", "spearman", "top_k_hit_rate"})
        self.assertEqual(result["top_k_hit_rate"], 1.0)

    def test_predict_with_fitness_head_returns_one_prediction_per_embedding(self):
        import tempfile
        from pathlib import Path

        import torch

        from aav_baseline.llm_agent_core import FitnessMLP

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            model = FitnessMLP()
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            predictions = predict_with_fitness_head([[0.0] * 480, [1.0] * 480], checkpoint_path)

        self.assertEqual(predictions.shape, (2,))

    def test_one_hot_encode_regions_uses_20_features_per_position(self):
        encoded = one_hot_encode_regions(["AC", "CA"])

        self.assertEqual(encoded.shape, (2, 40))
        self.assertEqual(encoded[0].sum(), 2)

    def test_score_mutation_rules_penalizes_large_mutation_sets(self):
        result = score_mutation_rules(["A1V"] * 12)

        self.assertLess(result["mutation_count_penalty"], 0)

    def test_knowledge_context_describes_mutation_properties(self):
        self.assertIn("D561A", build_knowledge_context(["D561A"]))
        self.assertIn("D", amino_acid_properties())

    def test_weighted_knowledge_score_normalizes_different_scales(self):
        df = pd.DataFrame(
            {
                "historical_prior": [10.0, 20.0, 30.0],
                "rule_score": [-4.0, 0.0, 4.0],
            }
        )

        result = weighted_knowledge_score(df, historical_weight=0.7, rule_weight=0.3)

        self.assertEqual(list(result["historical_prior_norm"]), [0.0, 0.5, 1.0])
        self.assertEqual(list(result["rule_score_norm"]), [0.0, 0.5, 1.0])
        self.assertEqual(list(result["knowledge_enhanced_score"].round(2)), [0.0, 0.5, 1.0])

    def test_min_max_normalize_handles_constant_values(self):
        result = min_max_normalize(pd.Series([3.0, 3.0]))

        self.assertEqual(list(result), [0.0, 0.0])

    def test_build_mutation_knowledge_graph_emits_expected_triples(self):
        variant = {"candidate_id": "C00001", "mutations": ["D561A"], "true_fitness": 1.2}

        triples = build_mutation_knowledge_graph([variant])

        self.assertIn(("A", "has_property", "hydrophobic"), triples)
        self.assertIn(("D561A", "occurs_at", "561"), triples)
        self.assertIn(("C00001", "contains", "D561A"), triples)
        self.assertIn(("D561A", "observed_with_fitness", "1.200"), triples)

    def test_apply_mutation_count_constraint_keeps_low_mutation_candidates(self):
        df = pd.DataFrame(
            [
                {"candidate_id": "low", "num_mutations": 4},
                {"candidate_id": "high", "num_mutations": 5},
            ]
        )

        result = apply_mutation_count_constraint(df, max_mutations=4)

        self.assertEqual(list(result["candidate_id"]), ["low"])

    def test_random_strategy_returns_requested_number_without_duplicates(self):
        result = select_random_candidates(["A", "B", "C"], top_k=2, seed=1)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(set(result)), 2)

    def test_model_strategy_sorts_by_predicted_fitness(self):
        candidates = pd.DataFrame(
            [
                {"candidate_id": "A", "predicted_fitness": 1.0},
                {"candidate_id": "B", "predicted_fitness": 3.0},
                {"candidate_id": "C", "predicted_fitness": 2.0},
            ]
        )

        self.assertEqual(select_model_top_candidates(candidates, top_k=2), ["B", "C"])

    def test_summarize_rounds_returns_one_row_per_strategy_round(self):
        rows = [
            {"strategy": "random", "round": 1, "fitness": 1.0},
            {"strategy": "random", "round": 1, "fitness": 3.0},
        ]

        result = summarize_rounds(rows)

        self.assertEqual(result.iloc[0]["mean_fitness"], 2.0)

    def test_iterative_evolution_starts_from_training_data_and_updates_history(self):
        observed = pd.DataFrame(
            [
                {"candidate_id": "train_a", "target": 1.0},
                {"candidate_id": "train_b", "target": 2.0},
            ]
        )
        candidates = pd.DataFrame(
            [
                {"candidate_id": "cand_a", "target": 3.0, "predicted_fitness": 0.2},
                {"candidate_id": "cand_b", "target": 4.0, "predicted_fitness": 0.9},
            ]
        )
        observed_sizes = []

        def strategy(current_observed, available_candidates, round_index, top_k):
            observed_sizes.append(len(current_observed))
            return [available_candidates.sort_values("predicted_fitness", ascending=False).iloc[0]["candidate_id"]]

        result = simulate_iterative_evolution(
            observed,
            candidates,
            {"model": strategy},
            rounds=2,
            top_k=1,
            evaluator_col="target",
        )

        self.assertEqual(observed_sizes, [2, 3])
        self.assertEqual(list(result["round"]), [0, 1, 2])
        self.assertEqual(list(result["candidate_id"]), ["initial_training_data", "cand_b", "cand_a"])
        self.assertEqual(list(result["fitness"]), [2.0, 4.0, 3.0])

    def test_feedback_driven_agent_uses_previous_round_success_and_failure(self):
        observed = pd.DataFrame(
            [
                {"candidate_id": "hist_good", "target": 8.0, "mutations": ["S578E"]},
                {"candidate_id": "hist_bad", "target": -5.0, "mutations": ["D561A"]},
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "keeps_success",
                    "mutations": ["S578D"],
                    "num_mutations": 1,
                    "predicted_fitness": 1.0,
                    "knowledge_enhanced_score": 0.1,
                },
                {
                    "candidate_id": "keeps_failure",
                    "mutations": ["D561V"],
                    "num_mutations": 1,
                    "predicted_fitness": 5.0,
                    "knowledge_enhanced_score": 1.0,
                },
            ]
        )

        strategy = build_feedback_driven_agent_strategy(max_mutations=4)
        selected = strategy(observed, candidates, round_index=2, top_k=1)

        self.assertEqual(selected, ["keeps_success"])

    def test_summarize_feedback_prefers_latest_virtual_round(self):
        observed = pd.DataFrame(
            [
                {"target": 10.0, "mutations": ["D561A"], "virtual_feedback_round": None},
                {"target": 1.0, "mutations": ["S578D"], "virtual_feedback_round": 1},
                {"target": 9.0, "mutations": ["T581E"], "virtual_feedback_round": 2},
            ]
        )

        feedback = summarize_feedback(observed)

        self.assertEqual(feedback["successful_positions"], [581])
        self.assertEqual(feedback["failed_positions"], [581])

    def test_openai_feedback_selector_uses_llm_selected_candidate_ids(self):
        class FakeResponses:
            def parse(self, **kwargs):
                class Parsed:
                    output_parsed = RoundCriticDecision(
                        selected=[
                            RoundCandidateDecision(candidate_id="B", recommendation_reason="LLM selected B"),
                        ],
                        critic_feedback="Previous round favored position 578.",
                        successful_patterns=["S578"],
                        failed_patterns=["D561"],
                        next_round_strategy="Select B.",
                        limitations="Fake test response.",
                    )

                return Parsed()

        class FakeClient:
            responses = FakeResponses()

        observed = pd.DataFrame(
            [
                {"candidate_id": "prev", "target": 8.0, "mutations": ["S578E"], "num_mutations": 1},
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "A",
                    "mutations": ["S578D"],
                    "num_mutations": 1,
                    "predicted_fitness": 9.0,
                    "knowledge_enhanced_score": 1.0,
                },
                {
                    "candidate_id": "B",
                    "mutations": ["T581E"],
                    "num_mutations": 1,
                    "predicted_fitness": 1.0,
                    "knowledge_enhanced_score": 0.1,
                },
            ]
        )

        selected_ids, decision, _ = select_openai_feedback_candidates(
            FakeClient(),
            "fake-model",
            observed,
            candidates,
            round_index=1,
            top_k=1,
            candidate_pool_size=2,
            max_mutations=4,
        )

        self.assertEqual(selected_ids, ["B"])
        self.assertIn("Previous round", decision.critic_feedback)

    def test_summarize_recommended_positions_counts_mutation_frequency_by_round(self):
        results = pd.DataFrame(
            [
                {"strategy": "agent", "round": 1, "mutations": ["D561A", "E562K"]},
                {"strategy": "agent", "round": 1, "mutations": ["D561V"]},
                {"strategy": "agent", "round": 2, "mutations": ["E562D"]},
            ]
        )

        summary = summarize_recommended_positions(results)

        row = summary[(summary["round"] == 1) & (summary["position"] == 561)].iloc[0]
        self.assertEqual(row["mutation_count"], 2)
        self.assertEqual(row["variant_count"], 2)
        self.assertEqual(row["mutation_frequency"], 1.0)

    def test_demo_recommendations_score_candidates_from_supplied_tables(self):
        observed = pd.DataFrame(
            [
                {"sequence": "P" * 560 + WT_REGION + "S" * 147, "target": 1.0},
                {"sequence": "P" * 560 + WT_REGION[:18] + "A" + WT_REGION[19:] + "S" * 147, "target": 3.0},
            ]
        )
        candidate = pd.DataFrame(
            [
                {
                    "candidate_id": "C00000",
                    "sequence": "P" * 560 + WT_REGION[:18] + "A" + WT_REGION[19:] + "S" * 147,
                    "target": 2.0,
                }
            ]
        )
        predictions = pd.DataFrame(
            [{"candidate_id": "C00000", "predicted_fitness": 2.5}]
        )

        result = recommend_mutations_for_region(
            WT_REGION,
            observed,
            candidate,
            predictions=predictions,
            top_k=1,
            max_mutations=4,
        )

        self.assertEqual(result.iloc[0]["candidate_id"], "C00000")
        self.assertEqual(result.iloc[0]["predicted_fitness"], 2.5)
        self.assertIn("mutation_count", result.iloc[0]["recommendation_reason"])

    def test_demo_recommendations_respect_max_mutations_strictly(self):
        observed = pd.DataFrame(
            [
                {"sequence": "P" * 560 + WT_REGION + "S" * 147, "target": 1.0},
                {"sequence": "P" * 560 + WT_REGION[:18] + "A" + WT_REGION[19:] + "S" * 147, "target": 3.0},
            ]
        )
        three_mutation_region = "A" + WT_REGION[1:5] + "A" + WT_REGION[6:18] + "A" + WT_REGION[19:]
        candidate = pd.DataFrame(
            [
                {
                    "candidate_id": "C00000",
                    "sequence": "P" * 560 + three_mutation_region + "S" * 147,
                    "target": 2.0,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "No candidates"):
            recommend_mutations_for_region(
                WT_REGION,
                observed,
                candidate,
                top_k=1,
                max_mutations=2,
            )


if __name__ == "__main__":
    unittest.main()
