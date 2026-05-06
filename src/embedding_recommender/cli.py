"""Command-line interface for the embedding strategy recommender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .intent import VALID_INTENT_CATEGORIES, validate_intent
from .recommender import run
from .reporter import write_all


def build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="embedding-recommender",
        description=(
            "Profiles a CSV geospatial dataset and recommends vector embedding "
            "strategies for a stated user intent."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--intent-category",
        required=True,
        choices=sorted(VALID_INTENT_CATEGORIES),
        help="The category of task the user wants to perform on the data.",
    )
    parser.add_argument(
        "--intent-description",
        required=True,
        type=str,
        help="A short free-text description of the specific task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./reports"),
        help="Directory where report.json and report.md will be written. "
             "Defaults to ./reports.",
    )
    parser.add_argument(
        "--rubric-version",
        type=str,
        default=config.RUBRIC_VERSION,
        help=f"Rubric version to apply. Defaults to {config.RUBRIC_VERSION}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a Unix-style exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        intent = validate_intent(args.intent_category, args.intent_description)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        api_key = config.get_api_key()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    model = config.get_model()

    try:
        report = run(
            input_path=args.input,
            intent=intent,
            api_key=api_key,
            model=model,
            rubric_version=args.rubric_version,
            tool_version=config.TOOL_VERSION,
            llm_provider_name="anthropic",
            temperature=config.LLM_TEMPERATURE,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error during recommendation pipeline: {e}", file=sys.stderr)
        return 1

    try:
        paths = write_all(report, args.output_dir)
    except Exception as e:
        print(f"Error writing report: {e}", file=sys.stderr)
        return 1

    print("Report written:")
    print(f"  JSON:     {paths['json']}")
    print(f"  Markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
