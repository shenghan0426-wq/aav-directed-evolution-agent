from __future__ import annotations

import re

import pandas as pd


def amino_acid_properties() -> dict[str, dict[str, str]]:
    return {
        "A": {"charge": "neutral", "polarity": "nonpolar", "size": "small", "hydrophobicity": "hydrophobic"},
        "C": {"charge": "neutral", "polarity": "polar", "size": "small", "hydrophobicity": "hydrophobic"},
        "D": {"charge": "negative", "polarity": "polar", "size": "small", "hydrophobicity": "hydrophilic"},
        "E": {"charge": "negative", "polarity": "polar", "size": "medium", "hydrophobicity": "hydrophilic"},
        "F": {"charge": "neutral", "polarity": "nonpolar", "size": "large", "hydrophobicity": "hydrophobic"},
        "G": {"charge": "neutral", "polarity": "nonpolar", "size": "small", "hydrophobicity": "neutral"},
        "H": {"charge": "positive", "polarity": "polar", "size": "medium", "hydrophobicity": "hydrophilic"},
        "I": {"charge": "neutral", "polarity": "nonpolar", "size": "medium", "hydrophobicity": "hydrophobic"},
        "K": {"charge": "positive", "polarity": "polar", "size": "large", "hydrophobicity": "hydrophilic"},
        "L": {"charge": "neutral", "polarity": "nonpolar", "size": "medium", "hydrophobicity": "hydrophobic"},
        "M": {"charge": "neutral", "polarity": "nonpolar", "size": "medium", "hydrophobicity": "hydrophobic"},
        "N": {"charge": "neutral", "polarity": "polar", "size": "small", "hydrophobicity": "hydrophilic"},
        "P": {"charge": "neutral", "polarity": "nonpolar", "size": "small", "hydrophobicity": "neutral"},
        "Q": {"charge": "neutral", "polarity": "polar", "size": "medium", "hydrophobicity": "hydrophilic"},
        "R": {"charge": "positive", "polarity": "polar", "size": "large", "hydrophobicity": "hydrophilic"},
        "S": {"charge": "neutral", "polarity": "polar", "size": "small", "hydrophobicity": "hydrophilic"},
        "T": {"charge": "neutral", "polarity": "polar", "size": "small", "hydrophobicity": "hydrophilic"},
        "V": {"charge": "neutral", "polarity": "nonpolar", "size": "small", "hydrophobicity": "hydrophobic"},
        "W": {"charge": "neutral", "polarity": "nonpolar", "size": "large", "hydrophobicity": "hydrophobic"},
        "Y": {"charge": "neutral", "polarity": "polar", "size": "large", "hydrophobicity": "hydrophobic"},
    }


CONSERVATIVE_GROUPS = [
    set("AVLIM"),
    set("FWY"),
    set("STNQ"),
    set("DE"),
    set("KRH"),
    set("GP"),
]


def parse_mutation(mutation: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation)
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def is_conservative(wt: str, mut: str) -> bool:
    return any(wt in group and mut in group for group in CONSERVATIVE_GROUPS)


def score_mutation_rules(mutations: list[str]) -> dict:
    properties = amino_acid_properties()
    invalid = []
    conservative = 0
    radical = 0
    charge_changes = 0

    for mutation in mutations:
        parsed = parse_mutation(mutation)
        if parsed is None:
            invalid.append(mutation)
            continue
        wt, _, mut = parsed
        if wt not in properties or mut not in properties:
            invalid.append(mutation)
            continue
        if is_conservative(wt, mut):
            conservative += 1
        else:
            radical += 1
        if properties[wt]["charge"] != properties[mut]["charge"]:
            charge_changes += 1

    count = len(mutations)
    mutation_count_penalty = 0 if count <= 8 else -(count - 8) * 0.5
    invalid_penalty = -2.0 * len(invalid)
    radical_penalty = -0.25 * radical
    score = conservative * 0.2 + mutation_count_penalty + invalid_penalty + radical_penalty

    return {
        "mutation_count": count,
        "conservative_count": conservative,
        "radical_count": radical,
        "charge_change_count": charge_changes,
        "invalid_mutations": invalid,
        "mutation_count_penalty": mutation_count_penalty,
        "rule_score": float(score),
    }


def build_knowledge_context(mutations: list[str]) -> str:
    properties = amino_acid_properties()
    summary = score_mutation_rules(mutations)
    lines = [
        "Knowledge-enhanced mutation review:",
        f"- mutation_count: {summary['mutation_count']}",
        f"- conservative_count: {summary['conservative_count']}",
        f"- radical_count: {summary['radical_count']}",
        f"- charge_change_count: {summary['charge_change_count']}",
        f"- rule_score: {summary['rule_score']:.2f}",
    ]
    for mutation in mutations:
        parsed = parse_mutation(mutation)
        if parsed is None:
            lines.append(f"- {mutation}: invalid mutation format")
            continue
        wt, position, mut = parsed
        if wt not in properties or mut not in properties:
            lines.append(f"- {mutation}: invalid amino acid")
            continue
        mode = "conservative" if is_conservative(wt, mut) else "radical"
        lines.append(
            f"- {mutation}: position {position}, {mode}, "
            f"{properties[wt]['charge']}->{properties[mut]['charge']}, "
            f"{properties[wt]['polarity']}->{properties[mut]['polarity']}"
        )
    return "\n".join(lines)


def min_max_normalize(values) -> pd.Series:
    series = pd.Series(values, dtype=float)
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - minimum) / (maximum - minimum)


def weighted_knowledge_score(
    candidates: pd.DataFrame,
    historical_weight: float = 0.7,
    rule_weight: float = 0.3,
) -> pd.DataFrame:
    if "historical_prior" not in candidates.columns or "rule_score" not in candidates.columns:
        raise ValueError("candidates must contain historical_prior and rule_score columns.")

    result = candidates.copy()
    result["historical_prior_norm"] = min_max_normalize(result["historical_prior"])
    result["rule_score_norm"] = min_max_normalize(result["rule_score"])
    result["knowledge_enhanced_score"] = (
        historical_weight * result["historical_prior_norm"]
        + rule_weight * result["rule_score_norm"]
    )
    return result


def apply_mutation_count_constraint(candidates: pd.DataFrame, max_mutations: int = 4) -> pd.DataFrame:
    if "num_mutations" not in candidates.columns:
        raise ValueError("candidates must contain a num_mutations column.")
    return candidates[candidates["num_mutations"] <= max_mutations].copy()


def build_mutation_knowledge_graph(variants) -> list[tuple[str, str, str]]:
    properties = amino_acid_properties()
    triples = []

    for amino_acid, attrs in properties.items():
        triples.append((amino_acid, "has_property", attrs["hydrophobicity"]))
        triples.append((amino_acid, "has_property", attrs["polarity"]))
        triples.append((amino_acid, "has_property", attrs["charge"]))
        triples.append((amino_acid, "has_size", attrs["size"]))

    for variant in variants:
        variant_id = str(variant.get("candidate_id", variant.get("variant_id", "unknown_variant")))
        mutations = variant.get("mutations", [])
        fitness = variant.get("true_fitness", variant.get("target"))

        for mutation in mutations:
            parsed = parse_mutation(str(mutation))
            triples.append((variant_id, "contains", str(mutation)))
            if parsed is None:
                continue
            wt, position, mut = parsed
            triples.append((str(mutation), "occurs_at", str(position)))
            triples.append((str(mutation), "from_amino_acid", wt))
            triples.append((str(mutation), "to_amino_acid", mut))
            triples.append(
                (
                    str(mutation),
                    "substitution_type",
                    "conservative" if is_conservative(wt, mut) else "radical",
                )
            )
            if fitness is not None and not pd.isna(fitness):
                triples.append((str(mutation), "observed_with_fitness", f"{float(fitness):.3f}"))

    return triples
