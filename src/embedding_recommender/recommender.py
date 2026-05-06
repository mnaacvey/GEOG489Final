"""Orchestrates the profiler, rubric, and Anthropic API into a complete report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import config
from .data_source import load_csv
from .llm_provider import generate_recommendations
from .profiler import profile as run_profiler
from .rubric import get_rubric


def run(
    input_path: str | Path,
    intent: Dict[str, Any],
    api_key: str,
    model: str,
    rubric_version: str = config.RUBRIC_VERSION,
    tool_version: str = config.TOOL_VERSION,
    llm_provider_name: str = "anthropic",
    temperature: float = config.LLM_TEMPERATURE,
) -> Dict[str, Any]:
    """Runs the full pipeline and returns the assembled report dict.

    Steps:
    1. Load the CSV.
    2. Profile the dataset.
    3. Build the rubric prompt and call the Anthropic API.
    4. Compute warnings.
    5. Assemble and return the final report dict.
    """
    df, source_metadata = load_csv(input_path)

    profile_dict = run_profiler(df)
    rubric_text = get_rubric(rubric_version)
    recommendations = generate_recommendations(
        profile=profile_dict,
        intent=intent,
        rubric_text=rubric_text,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    warnings = compute_warnings(df, profile_dict)

    report: Dict[str, Any] = {
        "report_metadata": {
            "tool_version": tool_version,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_path": source_metadata.get("source_path", "unknown"),
            "llm_provider": llm_provider_name,
            "llm_model": model,
            "rubric_version": rubric_version,
        },
        "dataset_profile": profile_dict,
        "user_intent": intent,
        "recommendations": recommendations,
        "warnings": warnings,
    }
    return report


def compute_warnings(df: pd.DataFrame, profile_dict: Dict[str, Any]) -> List[str]:
    """Returns a list of profiler-detected warnings.

    Engineering hygiene items the LLM might overlook: encoding issues,
    suspicious nulls, and unusual scale or extent characteristics.
    """
    warnings: List[str] = []

    text_cols = profile_dict["attribute_summary"].get("text_columns", [])
    for col in text_cols:
        sample = df[col].dropna().astype(str).head(100)
        if any(not s.isascii() for s in sample):
            warnings.append(
                f"Text column '{col}' contains non-ASCII characters. "
                f"Verify your tokenizer handles UTF-8 correctly."
            )
            break

    for col in df.columns:
        null_ratio = df[col].isna().sum() / max(1, len(df))
        if null_ratio > 0.5:
            warnings.append(
                f"Column '{col}' is {round(null_ratio * 100)}% null. "
                f"Consider whether to include it in the embedding input."
            )

    if profile_dict["row_count"] < 100:
        warnings.append(
            f"Dataset has only {profile_dict['row_count']} rows. Embedding "
            f"quality is hard to evaluate at this scale; consider validating "
            f"on a larger sample."
        )

    geom = profile_dict.get("geometry", {})
    bbox = geom.get("spatial_extent_bbox")
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        if (max_lat - min_lat) < 1.0 and (max_lon - min_lon) < 1.0:
            warnings.append(
                "Spatial extent is small (sub-degree). For coordinate "
                "normalization, use the bounding box of this dataset rather "
                "than a global range."
            )

    return warnings
