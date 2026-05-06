"""Tests for the reporter module."""

import json
from pathlib import Path

from embedding_recommender.reporter import write_all
from embedding_recommender.schema import CANDIDATE_STRATEGIES


def _make_valid_report() -> dict:
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
            "signal_rationale": {"text_richness": "rationale"},
        },
        "user_intent": {
            "intent_category": "similarity_search",
            "description": "find similar things",
        },
        "recommendations": [
            {
                "rank": i + 1,
                "strategy": strategy,
                "score": round(0.9 - 0.15 * i, 2),
                "rationale": "Refers to text_richness signal.",
                "implementation_notes": ["A note"] if i == 0 else [],
                "validation_suggestions": ["Validate this"] if i == 0 else [],
                "code_scaffolding_ref": f"scaffolds/{strategy}.py",
            }
            for i, strategy in enumerate(CANDIDATE_STRATEGIES)
        ],
        "warnings": [],
    }


def test_write_all_creates_both_files(tmp_path: Path):
    report = _make_valid_report()
    paths = write_all(report, tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()


def test_write_all_json_is_valid_and_complete(tmp_path: Path):
    report = _make_valid_report()
    paths = write_all(report, tmp_path)
    parsed = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert parsed["report_metadata"]["tool_version"] == "0.1.0"
    assert len(parsed["recommendations"]) == 5


def test_write_all_markdown_contains_key_sections(tmp_path: Path):
    report = _make_valid_report()
    paths = write_all(report, tmp_path)
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "# Embedding Strategy Recommendation Report" in md
    assert "## Dataset Profile" in md
    assert "## Recommendations" in md
    assert "Rank 1:" in md
