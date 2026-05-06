"""Tests for the profiler module."""

import pandas as pd

from embedding_recommender.profiler import (
    classify_columns,
    compute_signals,
    detect_geometry,
    profile,
)


def test_classify_columns_text_rich_dataset():
    """Text-rich dataset should produce text and identifier columns."""
    df = pd.DataFrame(
        {
            "osm_id": list(range(100)),
            "name": [f"Place {i}" for i in range(100)],
            "description": [
                "A long description that exceeds twenty chars for sure number "
                + str(i)
                for i in range(100)
            ],
            "category": ["cafe"] * 50 + ["park"] * 50,
            "latitude": [38.9 + i * 0.001 for i in range(100)],
            "longitude": [-77.0 - i * 0.001 for i in range(100)],
        }
    )
    classified = classify_columns(df)
    assert "name" in classified["text_columns"]
    assert "description" in classified["text_columns"]
    assert "category" in classified["categorical_columns"]
    assert "osm_id" in classified["identifier_columns"]
    # Coordinate columns must not appear in any classified bucket.
    for bucket in classified.values():
        assert "latitude" not in bucket
        assert "longitude" not in bucket


def test_detect_geometry_with_lat_lon():
    """Detects lat/lon columns and computes a bounding box."""
    df = pd.DataFrame(
        {
            "latitude": [38.9, 38.95, 39.0],
            "longitude": [-77.0, -77.05, -77.1],
        }
    )
    geom = detect_geometry(df)
    assert geom["lat_column"] == "latitude"
    assert geom["lon_column"] == "longitude"
    assert geom["implied_type"] == "point"
    assert geom["spatial_extent_bbox"][0] < geom["spatial_extent_bbox"][2]


def test_detect_geometry_without_coords():
    """Returns 'none' geometry when no lat/lon columns are found."""
    df = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    geom = detect_geometry(df)
    assert geom["implied_type"] == "none"
    assert geom["spatial_extent_bbox"] is None


def test_compute_signals_normalizes_to_unit_range():
    """All signal values must be in [0, 1] except scale_tier."""
    df = pd.DataFrame(
        {
            "name": ["alpha"] * 10,
            "category": ["x"] * 10,
            "latitude": [0.0] * 10,
            "longitude": [0.0] * 10,
        }
    )
    classified = classify_columns(df)
    geom = detect_geometry(df)
    signals = compute_signals(df, classified, geom)
    for key in [
        "text_richness",
        "categorical_density",
        "numeric_attribute_presence",
        "geometric_complexity",
    ]:
        assert 0.0 <= signals[key] <= 1.0, f"{key} out of range: {signals[key]}"
    assert signals["scale_tier"] in {"small", "medium", "large"}


def test_profile_full_pipeline_returns_complete_dict():
    """The top-level profile function returns all required schema fields."""
    df = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "category": ["x", "y", "x"],
            "value": [1.0, 2.0, 3.0],
            "latitude": [38.9, 38.95, 39.0],
            "longitude": [-77.0, -77.05, -77.1],
        }
    )
    p = profile(df)
    assert set(p.keys()) == {
        "row_count",
        "column_count",
        "geometry",
        "attribute_summary",
        "signals",
        "signal_rationale",
    }
    assert p["row_count"] == 3
    assert p["column_count"] == 5
