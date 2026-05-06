"""JSON schemas for validating the report and the LLM response.

Two schemas are exported:
- REPORT_SCHEMA: the full report written to disk.
- RECOMMENDATIONS_SCHEMA: just the recommendations array, used to validate
  the LLM response before assembling the full report.
"""

from typing import Any, Dict


CANDIDATE_STRATEGIES = [
    "text_attribute_embedding",
    "categorical_attribute_embedding",
    "geometric_embedding",
    "hybrid_text_geometric",
    "hybrid_categorical_geometric",
]


VALID_INTENT_CATEGORIES = [
    "similarity_search",
    "classification",
    "clustering",
    "rag_qa",
    "spatial_relational",
]


# Schema fragment describing a single recommendation entry. Reused by both
# of the schemas below.
_RECOMMENDATION_ITEM: Dict[str, Any] = {
    "type": "object",
    "required": [
        "rank",
        "strategy",
        "score",
        "rationale",
        "implementation_notes",
        "validation_suggestions",
        "code_scaffolding_ref",
    ],
    "properties": {
        "rank": {"type": "integer", "minimum": 1, "maximum": 5},
        "strategy": {"type": "string", "enum": CANDIDATE_STRATEGIES},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "minLength": 1},
        "implementation_notes": {"type": "array", "items": {"type": "string"}},
        "validation_suggestions": {"type": "array", "items": {"type": "string"}},
        "code_scaffolding_ref": {"type": "string", "minLength": 1},
    },
}


# Schema for the recommendations array on its own. Used when validating the
# LLM response before assembling the full report.
RECOMMENDATIONS_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "minItems": 5,
    "maxItems": 5,
    "items": _RECOMMENDATION_ITEM,
}


# Schema for the full report. Used when validating the assembled report
# before writing it to disk.
REPORT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "report_metadata",
        "dataset_profile",
        "user_intent",
        "recommendations",
        "warnings",
    ],
    "properties": {
        "report_metadata": {
            "type": "object",
            "required": [
                "tool_version",
                "generated_at",
                "input_path",
                "llm_provider",
                "llm_model",
                "rubric_version",
            ],
            "properties": {
                "tool_version": {"type": "string"},
                "generated_at": {"type": "string"},
                "input_path": {"type": "string"},
                "llm_provider": {"type": "string"},
                "llm_model": {"type": "string"},
                "rubric_version": {"type": "string"},
            },
        },
        "dataset_profile": {
            "type": "object",
            "required": [
                "row_count",
                "column_count",
                "geometry",
                "attribute_summary",
                "signals",
                "signal_rationale",
            ],
            "properties": {
                "row_count": {"type": "integer", "minimum": 0},
                "column_count": {"type": "integer", "minimum": 0},
                "geometry": {"type": "object"},
                "attribute_summary": {"type": "object"},
                "signals": {
                    "type": "object",
                    "required": [
                        "text_richness",
                        "categorical_density",
                        "numeric_attribute_presence",
                        "geometric_complexity",
                        "scale_tier",
                    ],
                    "properties": {
                        "text_richness": {"type": "number", "minimum": 0, "maximum": 1},
                        "categorical_density": {"type": "number", "minimum": 0, "maximum": 1},
                        "numeric_attribute_presence": {"type": "number", "minimum": 0, "maximum": 1},
                        "geometric_complexity": {"type": "number", "minimum": 0, "maximum": 1},
                        "scale_tier": {"type": "string", "enum": ["small", "medium", "large"]},
                    },
                },
                "signal_rationale": {"type": "object"},
            },
        },
        "user_intent": {
            "type": "object",
            "required": ["intent_category", "description"],
            "properties": {
                "intent_category": {"type": "string", "enum": VALID_INTENT_CATEGORIES},
                "description": {"type": "string", "minLength": 1},
            },
        },
        "recommendations": RECOMMENDATIONS_SCHEMA,
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_recommendations_business_rules(recommendations: list) -> None:
    """Enforces business rules that JSON Schema cannot easily express.

    Specifically: rank values must be a permutation of [1, 2, 3, 4, 5], and
    the strategy values must each appear exactly once.

    Raises ValueError if a rule is violated.
    """
    ranks = sorted(item["rank"] for item in recommendations)
    if ranks != [1, 2, 3, 4, 5]:
        raise ValueError(f"Ranks must be a permutation of 1-5, got {ranks}.")

    strategies = sorted(item["strategy"] for item in recommendations)
    if strategies != sorted(CANDIDATE_STRATEGIES):
        raise ValueError(
            f"Each candidate strategy must appear exactly once. Got {strategies}."
        )
