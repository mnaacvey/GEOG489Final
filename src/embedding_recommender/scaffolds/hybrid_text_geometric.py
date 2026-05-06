"""Starter skeleton for the hybrid_text_geometric strategy.

Concatenates an L2-normalized text embedding with normalized coordinates
scaled by spatial_weight. Fits when text_richness is high and spatial
proximity matters for similarity (e.g. "similar things nearby").
"""

from typing import Iterable
import numpy as np
import pandas as pd

from .text_attribute_embedding import build_embeddings as build_text_embeddings
from .geometric_embedding import build_embeddings as build_geometric_embeddings


def build_embeddings(
    df: pd.DataFrame,
    text_columns: Iterable[str],
    lat_column: str,
    lon_column: str,
    spatial_weight: float = 0.3,
) -> np.ndarray:
    """Returns shape (n_rows, text_dim + 2). Wire up after constituent scaffolds work."""
    # TODO: text_part = build_text_embeddings(df, text_columns)
    # TODO: geom_part = build_geometric_embeddings(df, lat_column, lon_column)
    # TODO: return np.concatenate([text_part, spatial_weight * geom_part], axis=1)
    raise NotImplementedError("Wire this up after text and geometric scaffolds are filled in.")