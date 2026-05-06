"""Loads input data into a pandas DataFrame."""

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Reads a CSV and returns (DataFrame, metadata dict)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return pd.read_csv(p), {"source_path": str(p), "format": "csv"}