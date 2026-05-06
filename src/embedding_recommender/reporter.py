"""Validates the report and writes JSON and Markdown output files.

The Markdown renderer is plain Python f-strings. No templating engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import validate

from .schema import REPORT_SCHEMA, validate_recommendations_business_rules


def validate_report(report: Dict[str, Any]) -> None:
    """Validates the report against the schema and the business rules.

    Raises a ValidationError or ValueError on failure.
    """
    validate(instance=report, schema=REPORT_SCHEMA)
    validate_recommendations_business_rules(report["recommendations"])


def write_json(report: Dict[str, Any], output_dir: Path) -> Path:
    """Writes the report as pretty-printed JSON. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def write_markdown(report: Dict[str, Any], output_dir: Path) -> Path:
    """Renders the report as Markdown. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(_render_markdown(report))
    return path


def write_all(report: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """Validates the report and writes both formats. Returns a dict of paths."""
    validate_report(report)
    return {
        "json": write_json(report, output_dir),
        "markdown": write_markdown(report, output_dir),
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    """Renders the full report as a Markdown string."""
    parts = []
    parts.append(_render_header(report))
    parts.append(_render_metadata(report["report_metadata"]))
    parts.append(_render_profile(report["dataset_profile"]))
    parts.append(_render_intent(report["user_intent"]))
    parts.append(_render_recommendations(report["recommendations"]))
    parts.append(_render_warnings(report["warnings"]))
    return "\n\n".join(parts) + "\n"


def _render_header(report: Dict[str, Any]) -> str:
    return "# Embedding Strategy Recommendation Report"


def _render_metadata(metadata: Dict[str, Any]) -> str:
    return (
        "## Report Metadata\n\n"
        f"- **Generated at:** {metadata['generated_at']}\n"
        f"- **Input file:** `{metadata['input_path']}`\n"
        f"- **Tool version:** {metadata['tool_version']}\n"
        f"- **Rubric version:** {metadata['rubric_version']}\n"
        f"- **LLM provider:** {metadata['llm_provider']}\n"
        f"- **LLM model:** {metadata['llm_model']}"
    )


def _render_profile(profile: Dict[str, Any]) -> str:
    geom = profile["geometry"]
    bbox = geom.get("spatial_extent_bbox")
    bbox_str = (
        f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"
        if bbox
        else "none"
    )

    attrs = profile["attribute_summary"]
    signals = profile["signals"]
    rationale = profile["signal_rationale"]

    lines = [
        "## Dataset Profile",
        "",
        f"- **Rows:** {profile['row_count']:,}",
        f"- **Columns:** {profile['column_count']}",
        "",
        "### Geometry",
        "",
        f"- **Source:** {geom['source']}",
        f"- **Implied type:** {geom['implied_type']}",
        f"- **Bounding box:** {bbox_str}",
        f"- **Point density (per km^2):** {geom['point_density_per_km2']}",
        "",
        "### Attribute Summary",
        "",
        f"- **Text columns:** {_format_list(attrs.get('text_columns', []))}",
        f"- **Categorical columns:** {_format_list(attrs.get('categorical_columns', []))}",
        f"- **Numeric columns:** {_format_list(attrs.get('numeric_columns', []))}",
        f"- **Identifier columns:** {_format_list(attrs.get('identifier_columns', []))}",
        "",
        "### Signals",
        "",
        f"- **Text richness:** {signals['text_richness']}",
        f"- **Categorical density:** {signals['categorical_density']}",
        f"- **Numeric attribute presence:** {signals['numeric_attribute_presence']}",
        f"- **Geometric complexity:** {signals['geometric_complexity']}",
        f"- **Scale tier:** {signals['scale_tier']}",
        "",
        "### Signal Rationale",
        "",
    ]
    for key, value in rationale.items():
        lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


def _render_intent(intent: Dict[str, Any]) -> str:
    return (
        "## User Intent\n\n"
        f"- **Category:** {intent['intent_category']}\n"
        f"- **Description:** {intent['description']}"
    )


def _render_recommendations(recommendations: list) -> str:
    sorted_recs = sorted(recommendations, key=lambda r: r["rank"])
    lines = ["## Recommendations", ""]
    for rec in sorted_recs:
        lines.append(f"### Rank {rec['rank']}: {rec['strategy']} (score: {rec['score']})")
        lines.append("")
        lines.append(f"**Rationale:** {rec['rationale']}")
        lines.append("")
        if rec.get("implementation_notes"):
            lines.append("**Implementation notes:**")
            lines.append("")
            for note in rec["implementation_notes"]:
                lines.append(f"- {note}")
            lines.append("")
        if rec.get("validation_suggestions"):
            lines.append("**Validation suggestions:**")
            lines.append("")
            for suggestion in rec["validation_suggestions"]:
                lines.append(f"- {suggestion}")
            lines.append("")
        lines.append(f"**Code scaffolding:** `{rec['code_scaffolding_ref']}`")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_warnings(warnings: list) -> str:
    if not warnings:
        return "## Warnings\n\n*No warnings.*"
    lines = ["## Warnings", ""]
    for w in warnings:
        lines.append(f"- {w}")
    return "\n".join(lines)


def _format_list(items: list) -> str:
    """Formats a list as a comma-separated string of backticked names."""
    if not items:
        return "*none*"
    return ", ".join(f"`{item}`" for item in items)
