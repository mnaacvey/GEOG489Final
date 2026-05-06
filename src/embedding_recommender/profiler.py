"""Computes the dataset profile and signals from a pandas DataFrame.

The profiler is pure deterministic Python. The LLM consumes this output as
input; it does not produce it. All numeric signals are normalized to [0, 1]
so the rubric can reason about thresholds consistently regardless of dataset
size or shape.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from .timing import timed


# Substring lists used to classify columns by name. Matched as lowercase
# substrings against the lowercase column name. Easy to extend.
NAME_SUBSTRINGS = ["name", "title", "label"]
DESCRIPTION_SUBSTRINGS = ["description", "desc", "summary", "notes", "comment"]
TEXT_NAME_SUBSTRINGS = NAME_SUBSTRINGS + DESCRIPTION_SUBSTRINGS
ID_SUBSTRINGS = ["id", "uuid", "key", "guid", "code"]
LAT_SUBSTRINGS = ["latitude", "lat"]
LON_SUBSTRINGS = ["longitude", "lng", "lon"]

# Threshold above which we use multiprocessing for column classification.
# Below this we run single-process to avoid pool startup overhead.
PARALLEL_COLUMN_THRESHOLD = 50

# Geometric complexity lookup. Points have minimal geometric structure;
# polygons have the most.
GEOMETRIC_COMPLEXITY_BY_TYPE = {
    "none": 0.0,
    "point": 0.05,
    "line": 0.4,
    "polygon": 0.7,
    "mixed": 0.6,
}

# Scale tier thresholds.
SMALL_TIER_MAX_ROWS = 10_000
MEDIUM_TIER_MAX_ROWS = 1_000_000


def _name_matches_any(col_name: str, substrings: List[str]) -> bool:
    """Returns True if any of the substrings appears in the lowercased column name."""
    lowered = col_name.lower()
    return any(token in lowered for token in substrings)


def _is_coordinate_column(col_name: str) -> bool:
    """Returns True if the column name looks like a coordinate column.

    Used to exclude lat/lon columns from the numeric_columns classification.
    The check is intentionally narrow: we only match exact-token equivalents
    or names that contain 'latitude' or 'longitude'.
    """
    lowered = col_name.lower().strip()
    if lowered in {"lat", "latitude", "y", "lon", "lng", "longitude", "x"}:
        return True
    if "latitude" in lowered or "longitude" in lowered:
        return True
    return False


def _classify_one_column(args: Tuple[str, pd.Series, int]) -> Tuple[str, str]:
    """Classifies a single column. Returns (column_name, category).

    Categories: 'text', 'categorical', 'numeric', 'identifier', or 'other'.
    Lives at module level so it can be pickled for multiprocessing.
    """
    col_name, series, total_rows = args
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / max(1, total_rows)

    # Identifier columns: high cardinality and an ID-like name. Order matters:
    # check this before the numeric check, since ID columns are often integers.
    if unique_ratio > 0.9 and _name_matches_any(col_name, ID_SUBSTRINGS):
        return col_name, "identifier"

    # Coordinate columns are excluded from all categories. They are handled
    # by detect_geometry instead.
    if _is_coordinate_column(col_name):
        return col_name, "other"

    # Numeric columns: anything pandas thinks is numeric and is not a coord.
    if is_numeric_dtype(series):
        return col_name, "numeric"

    # String dtype: now split between text and categorical.
    if is_string_dtype(series):
        # Compute average non-null string length. A long average length
        # combined with many unique values suggests free text.
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return col_name, "other"

        avg_length = non_null.str.len().mean()
        long_strings = avg_length > 20
        name_looks_text = _name_matches_any(col_name, TEXT_NAME_SUBSTRINGS)

        # Text classification. We accept either a long average length or a
        # name-based hint. Either signal is enough on its own.
        if long_strings or name_looks_text:
            return col_name, "text"

        # Categorical classification. Low cardinality is the test, with both
        # an absolute and relative threshold so it works at multiple scales.
        if unique_ratio < 0.05 or n_unique < 50:
            return col_name, "categorical"

        # Fallthrough: a string column that does not match any category.
        return col_name, "other"

    return col_name, "other"


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classifies every column in the DataFrame into one of the schema categories.

    Returns a dict with keys text_columns, categorical_columns, numeric_columns,
    identifier_columns. Columns that do not fit any category are dropped.

    Uses multiprocessing when the column count is high. The threshold is set
    to avoid pool startup overhead for small inputs.
    """
    total_rows = len(df)
    work = [(col, df[col], total_rows) for col in df.columns]

    if len(work) > PARALLEL_COLUMN_THRESHOLD:
        with mp.Pool() as pool:
            results = pool.map(_classify_one_column, work)
    else:
        results = [_classify_one_column(item) for item in work]

    grouped: Dict[str, List[str]] = {
        "text_columns": [],
        "categorical_columns": [],
        "numeric_columns": [],
        "identifier_columns": [],
    }
    for col_name, category in results:
        if category == "text":
            grouped["text_columns"].append(col_name)
        elif category == "categorical":
            grouped["categorical_columns"].append(col_name)
        elif category == "numeric":
            grouped["numeric_columns"].append(col_name)
        elif category == "identifier":
            grouped["identifier_columns"].append(col_name)
    return grouped


def detect_geometry(df: pd.DataFrame) -> Dict[str, Any]:
    """Detects geometry information from the DataFrame.

    For CSV input we look for latitude and longitude columns. If found and
    numeric, we compute a point bounding box and basic density signals.
    If not found, we return a minimal dict indicating no geometry.
    """
    lat_col = None
    lon_col = None

    # Find the first column whose name matches a latitude or longitude pattern.
    # We prefer exact matches but fall back to substring matches.
    for col in df.columns:
        lowered = col.lower().strip()
        if lat_col is None:
            if lowered in {"lat", "latitude", "y"} or any(
                s in lowered for s in LAT_SUBSTRINGS
            ):
                if is_numeric_dtype(df[col]):
                    lat_col = col
        if lon_col is None:
            if lowered in {"lon", "lng", "longitude", "x"} or any(
                s in lowered for s in LON_SUBSTRINGS
            ):
                if is_numeric_dtype(df[col]):
                    lon_col = col

    if lat_col is None or lon_col is None:
        return {
            "source": "none",
            "lat_column": None,
            "lon_column": None,
            "implied_type": "none",
            "spatial_extent_bbox": None,
            "point_density_per_km2": 0.0,
        }

    # Compute the bounding box from non-null coordinate pairs.
    valid = df[[lat_col, lon_col]].dropna()
    if len(valid) == 0:
        return {
            "source": "lat_lon_columns",
            "lat_column": lat_col,
            "lon_column": lon_col,
            "implied_type": "none",
            "spatial_extent_bbox": None,
            "point_density_per_km2": 0.0,
        }

    min_lat = float(valid[lat_col].min())
    max_lat = float(valid[lat_col].max())
    min_lon = float(valid[lon_col].min())
    max_lon = float(valid[lon_col].max())
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # Approximate point density. We use a simple equirectangular conversion
    # at the bbox center latitude. This is good enough for a coarse density
    # signal; the rubric does not need precise area calculations.
    density = _approximate_density(valid, lat_col, lon_col, bbox)

    return {
        "source": "lat_lon_columns",
        "lat_column": lat_col,
        "lon_column": lon_col,
        "implied_type": "point",
        "spatial_extent_bbox": bbox,
        "point_density_per_km2": density,
    }


def _approximate_density(
    valid: pd.DataFrame, lat_col: str, lon_col: str, bbox: List[float]
) -> float:
    """Approximates point density per square kilometer using an equirectangular
    projection at the bbox center latitude.

    Returns zero if the bbox is degenerate (single point or zero area).
    """
    import math

    min_lon, min_lat, max_lon, max_lat = bbox
    if max_lat == min_lat or max_lon == min_lon:
        return 0.0

    # Latitude degrees are roughly 111 km each. Longitude degrees shrink with
    # the cosine of latitude.
    center_lat = (min_lat + max_lat) / 2.0
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))

    height_km = (max_lat - min_lat) * km_per_deg_lat
    width_km = (max_lon - min_lon) * km_per_deg_lon
    area_km2 = max(0.0001, height_km * width_km)

    return round(len(valid) / area_km2, 2)


def compute_signals(
    df: pd.DataFrame,
    columns_dict: Dict[str, List[str]],
    geometry_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes the five normalized profiler signals.

    Returns a dict matching the 'signals' section of the schema.
    """
    total_columns = max(1, len(df.columns))

    text_richness = _compute_text_richness(df, columns_dict["text_columns"], total_columns)
    categorical_density = round(len(columns_dict["categorical_columns"]) / total_columns, 2)
    numeric_presence = round(len(columns_dict["numeric_columns"]) / total_columns, 2)
    geometric_complexity = GEOMETRIC_COMPLEXITY_BY_TYPE.get(
        geometry_dict.get("implied_type", "none"), 0.0
    )
    scale_tier = _compute_scale_tier(len(df))

    return {
        "text_richness": text_richness,
        "categorical_density": categorical_density,
        "numeric_attribute_presence": numeric_presence,
        "geometric_complexity": geometric_complexity,
        "scale_tier": scale_tier,
    }


def _compute_text_richness(
    df: pd.DataFrame, text_columns: List[str], total_columns: int
) -> float:
    """Computes a weighted text richness score in [0, 1].

    The score combines three components:
    - column fraction: how much of the schema is text
    - average length: how long the text values are
    - lexical diversity: ratio of unique tokens to total tokens

    Each component is normalized to [0, 1] and the three are averaged.
    """
    if not text_columns:
        return 0.0

    column_fraction = len(text_columns) / total_columns

    # Concatenate all text values across the text columns to compute average
    # length and lexical diversity. We skip nulls.
    all_text = []
    for col in text_columns:
        non_null = df[col].dropna().astype(str).tolist()
        all_text.extend(non_null)

    if not all_text:
        return round(column_fraction / 3.0, 2)

    # Average length component. Saturates at 200 characters.
    avg_length = sum(len(t) for t in all_text) / len(all_text)
    length_component = min(1.0, avg_length / 200.0)

    # Lexical diversity component. We split on whitespace, which is fine for
    # English-like text and good enough for the MVP.
    tokens = []
    for text in all_text:
        tokens.extend(text.split())
    if not tokens:
        diversity_component = 0.0
    else:
        unique_tokens = len(set(t.lower() for t in tokens))
        diversity_component = unique_tokens / len(tokens)

    score = (column_fraction + length_component + diversity_component) / 3.0
    return round(score, 2)


def _compute_scale_tier(row_count: int) -> str:
    """Maps a row count to a discrete scale tier."""
    if row_count < SMALL_TIER_MAX_ROWS:
        return "small"
    if row_count < MEDIUM_TIER_MAX_ROWS:
        return "medium"
    return "large"


def compute_signal_rationale(
    df: pd.DataFrame,
    columns_dict: Dict[str, List[str]],
    geometry_dict: Dict[str, Any],
    signals: Dict[str, Any],
) -> Dict[str, str]:
    """Returns plain-language explanations of how each signal was computed."""
    rationales: Dict[str, str] = {}

    # Text richness rationale: report the number of text columns and the
    # average length of their content if any.
    text_cols = columns_dict["text_columns"]
    if text_cols:
        all_text = []
        for col in text_cols:
            all_text.extend(df[col].dropna().astype(str).tolist())
        if all_text:
            avg_length = round(sum(len(t) for t in all_text) / len(all_text))
            rationales["text_richness"] = (
                f"{len(text_cols)} text column(s) detected with mean length "
                f"{avg_length} characters."
            )
        else:
            rationales["text_richness"] = (
                f"{len(text_cols)} text column(s) detected but no non-null values."
            )
    else:
        rationales["text_richness"] = "No text columns detected."

    # Categorical density rationale.
    cat_cols = columns_dict["categorical_columns"]
    rationales["categorical_density"] = (
        f"{len(cat_cols)} categorical column(s) out of {len(df.columns)} total."
    )

    # Numeric presence rationale.
    num_cols = columns_dict["numeric_columns"]
    rationales["numeric_attribute_presence"] = (
        f"{len(num_cols)} numeric column(s) out of {len(df.columns)} total "
        f"(coordinate columns excluded)."
    )

    # Geometric complexity rationale.
    geom_type = geometry_dict.get("implied_type", "none")
    if geom_type == "none":
        rationales["geometric_complexity"] = "No geometry detected."
    else:
        rationales["geometric_complexity"] = (
            f"Geometry type {geom_type} detected via "
            f"{geometry_dict.get('source', 'unknown')}."
        )

    return rationales


@timed
def profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Runs the full profiler and returns the dataset_profile dict.

    Output matches the dataset_profile section of the schema.
    """
    columns_dict = classify_columns(df)
    geometry_dict = detect_geometry(df)
    signals = compute_signals(df, columns_dict, geometry_dict)
    rationales = compute_signal_rationale(df, columns_dict, geometry_dict, signals)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "geometry": geometry_dict,
        "attribute_summary": columns_dict,
        "signals": signals,
        "signal_rationale": rationales,
    }
