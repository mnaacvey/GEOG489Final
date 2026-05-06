"""Starter skeleton for the text_attribute_embedding strategy.

Embeds concatenated text columns using a sentence-transformer. Fits when
text_richness is high and intent is semantic (similarity_search, rag_qa,
classification on text-like targets).
"""

from typing import Iterable
import numpy as np
import pandas as pd


def build_embeddings(df: pd.DataFrame, text_columns: Iterable[str]) -> np.ndarray:
    """Returns shape (n_rows, embedding_dim). Replace TODO with a real encoder."""
    texts = df[list(text_columns)].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    # TODO: from sentence_transformers import SentenceTransformer
    # TODO: return SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
    raise NotImplementedError("Plug in your encoder above.")