"""Tests for the schema module."""

import pytest
from jsonschema import ValidationError, validate

from embedding_recommender.schema import (
    CANDIDATE_STRATEGIES,
    RECOMMENDATIONS_SCHEMA,
    REPORT_SCHEMA,
    validate_recommendations_business_rules,
)


def _make_valid_recommendations() -> list:
    """Helper that returns a valid recommendations array for tests."""
    return [
        {
            "rank": i + 1,
            "strategy": strategy,
            "score": round(0.9 - 0.15 * i, 2),
            "rationale": "References text_richness and categorical_density signals.",
            "implementation_notes": [],
            "validation_suggestions": [],
            "code_scaffolding_ref": f"scaffolds/{strategy}.py",
        }
        for i, strategy in enumerate(CANDIDATE_STRATEGIES)
    ]


def _make_valid_report() -> dict:
    """Helper that returns a valid full report for tests."""
    return {
        "report_metadata": {
            "tool_version": "0.1.0",
            "generated_at": "2026-04-25T14:32:00Z",
            "input_path": "data/sample.csv",
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4-5",
            "rubric_version": "mvp-modality-text-v1",
        },
        "dataset_profile": {
            "row_count": 100,
            "column_count": 5,
            "geometry": {
                "source": "lat_lon_columns",
                "lat_column": "latitude",
                "lon_column": "longitude",
                "implied_type": "point",
                "spatial_extent_bbox": [-77.1, 38.9, -77.0, 39.0],
                "point_density_per_km2": 50.0,
            },
            "attribute_summary": {
                "text_columns": ["name"],
                "categorical_columns": ["category"],
                "numeric_columns": ["value"],
                "identifier_columns": ["osm_id"],
            },
            "signals": {
                "text_richness": 0.6,
                "categorical_density": 0.2,
                "numeric_attribute_presence": 0.2,
                "geometric_complexity": 0.05,
                "scale_tier": "small",
            },
            "signal_rationale": {
                "text_richness": "rationale text",
            },
        },
        "user_intent": {
            "intent_category": "similarity_search",
            "description": "find similar things",
        },
        "recommendations": _make_valid_recommendations(),
        "warnings": [],
    }


def test_recommendations_schema_accepts_valid_array():
    validate(instance=_make_valid_recommendations(), schema=RECOMMENDATIONS_SCHEMA)


def test_recommendations_schema_rejects_wrong_count():
    recs = _make_valid_recommendations()[:4]
    with pytest.raises(ValidationError):
        validate(instance=recs, schema=RECOMMENDATIONS_SCHEMA)


def test_recommendations_business_rules_reject_duplicate_ranks():
    recs = _make_valid_recommendations()
    recs[1]["rank"] = recs[0]["rank"]
    with pytest.raises(ValueError, match="permutation"):
        validate_recommendations_business_rules(recs)


def test_report_schema_accepts_valid_report():
    validate(instance=_make_valid_report(), schema=REPORT_SCHEMA)


def test_report_schema_rejects_invalid_intent_category():
    report = _make_valid_report()
    report["user_intent"]["intent_category"] = "not_a_category"
    with pytest.raises(ValidationError):
        validate(instance=report, schema=REPORT_SCHEMA)
