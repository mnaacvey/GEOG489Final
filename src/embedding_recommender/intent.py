"""Validates the user intent input from the CLI."""

from typing import Dict


VALID_INTENT_CATEGORIES = {
    "similarity_search",
    "classification",
    "clustering",
    "rag_qa",
    "spatial_relational",
}


def validate_intent(category: str, description: str) -> Dict[str, str]:
    """Validates the intent inputs and returns the user_intent dict.

    Raises a ValueError with a clear message if the inputs are invalid.
    """
    if category not in VALID_INTENT_CATEGORIES:
        valid = ", ".join(sorted(VALID_INTENT_CATEGORIES))
        raise ValueError(
            f"Invalid intent_category '{category}'. Must be one of: {valid}."
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError("intent_description must be a non-empty string.")

    return {
        "intent_category": category,
        "description": description.strip(),
    }
