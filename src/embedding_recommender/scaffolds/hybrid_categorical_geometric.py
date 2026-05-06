"""Starter skeleton for the hybrid_categorical_geometric strategy.

Concatenates a categorical embedding with normalized coordinates. Fits when
both categorical_density and geometric_complexity are at least moderate and
intent is clustering or spatial_relational over categorical regions.
"""

from typing import Iterable
import numpy as np
import pandas as pd

from .categorical_attribute_embedding import build_embeddings as build_cat_embeddings
from .geometric_embedding import build_embeddings as build_geom_embeddings


def build_embeddings(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    lat_column: str,
    lon_column: str,
    numeric_columns: Iterable[str] | None = None,
    spatial_weight: float = 0.3,
) -> np.ndarray:
    """Returns shape (n_rows, cat_dim + 2). Wire up after constituent scaffolds work."""
    # TODO: cat_part = build_cat_embeddings(df, categorical_columns, numeric_columns)
    # TODO: geom_part = build_geom_embeddings(df, lat_column, lon_column)
    # TODO: return np.concatenate([cat_part, spatial_weight * geom_part], axis=1)
    raise NotImplementedError("Wire this up after categorical and geometric scaffolds are filled in.")