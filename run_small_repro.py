from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from app_demo import recommend_mutations_for_region
from llm_agent_core import WT_REGION, extract_region
from model_evaluation import one_hot_encode_regions


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
ARTIFACTS = BASE_DIR / "artifacts"


def load_sample_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SAMPLE_DIR / "sample_train.csv")
    validation = pd.read_csv(SAMPLE_DIR / "sample_validation.csv")
    test = pd.read_csv(SAMPLE_DIR / "sample_test.csv")
    return train, validation, test


def add_regions(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["mutated_region"] = result["sequence"].apply(extract_region)
    return result[result["mutated_region"].str.len() == len(WT_REGION)].reset_index(drop=True)


def train_ridge_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> Ridge:
    x_train = one_hot_encode_regions(train["mutated_region"])
    x_val = one_hot_encode_regions(validation["mutated_region"])
    model = Ridge(alpha=1.0)
    model.fit(x_train, train["target"])
    val_pred = model.predict(x_val)
    print(f"Validation MSE: {mean_squared_error(validation['target'], val_pred):.3f}")
    return model


def score_test_candidates(model: Ridge, test: pd.DataFrame) -> pd.DataFrame:
    x_test = one_hot_encode_regions(test["mutated_region"])
    predictions = pd.DataFrame(
        {
            "candidate_id": [f"C{i:05d}" for i in range(len(test))],
            "predicted_fitness": model.predict(x_test),
        }
    )
    return predictions


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    train, validation, test = load_sample_splits()
    train = add_regions(train)
    validation = add_regions(validation)
    test = add_regions(test)

    model = train_ridge_baseline(train, validation)
    predictions = score_test_candidates(model, test)

    recommendations = recommend_mutations_for_region(
        WT_REGION,
        train,
        test,
        predictions=predictions,
        top_k=10,
        max_mutations=4,
    )
    output_path = ARTIFACTS / "small_repro_recommendations.csv"
    recommendations.to_csv(output_path, index=False)

    print("\nTop recommendations:")
    print(
        recommendations[
            [
                "candidate_id",
                "mutations",
                "num_mutations",
                "predicted_fitness",
                "true_fitness",
                "knowledge_enhanced_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
