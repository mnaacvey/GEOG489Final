"""Tests for the intent module."""

import pytest

from embedding_recommender.intent import validate_intent


def test_validate_intent_accepts_valid_input():
    result = validate_intent("similarity_search", "Find similar coffee shops")
    assert result["intent_category"] == "similarity_search"
    assert result["description"] == "Find similar coffee shops"


def test_validate_intent_rejects_unknown_category():
    with pytest.raises(ValueError, match="Invalid intent_category"):
        validate_intent("not_a_real_category", "some description")


def test_validate_intent_rejects_empty_description():
    with pytest.raises(ValueError, match="non-empty"):
        validate_intent("clustering", "")


def test_validate_intent_strips_whitespace_in_description():
    result = validate_intent("clustering", "  group similar features  ")
    assert result["description"] == "group similar features"
