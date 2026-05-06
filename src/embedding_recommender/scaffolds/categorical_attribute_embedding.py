"""Starter skeleton for the categorical_attribute_embedding strategy.

One-hot or hashing encoders over categorical columns, optionally
concatenated with normalized numerics. Fits when categorical_density is
high and intent involves grouping or labeling (clustering, classification).
"""

from typing import Iterable
import numpy as np
import pandas as pd


def build_embeddings(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    numeric_columns: Iterable[str] | None = None,
) -> np.ndarray:
    """Returns shape (n_rows, embedding_dim). Replace TODO with a real encoder."""
    # TODO: from sklearn.preprocessing import OneHotEncoder, StandardScaler
    # TODO: ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    # TODO: cat_part = ohe.fit_transform(df[list(categorical_columns)].fillna("missing").astype(str))
    # TODO: if numeric_columns: concat with StandardScaler-normalized numerics
    raise NotImplementedError("Plug in your encoder above.")