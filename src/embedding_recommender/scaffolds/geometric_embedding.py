"""Starter skeleton for the geometric_embedding strategy.

Coordinate-based features. For points, normalized lat/lon. For lines and
polygons, extend with centroid, bbox, and geometry-type one-hot. Fits when
geometric_complexity is high or intent is spatial_relational.
"""

import numpy as np
import pandas as pd


def build_embeddings(df: pd.DataFrame, lat_column: str, lon_column: str) -> np.ndarray:
    """Returns shape (n_rows, 2) with coordinates normalized to the dataset bbox."""
    coords = df[[lat_column, lon_column]].dropna().to_numpy(dtype=float)
    if coords.shape[0] == 0:
        return np.zeros((0, 2))
    min_vals, max_vals = coords.min(axis=0), coords.max(axis=0)
    range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    return (coords - min_vals) / range_vals
    # TODO: extend with geometry-type one-hot, centroid, bbox dimensions for non-point data.