"""Loads environment variables and exposes module-level constants."""

import os

from dotenv import load_dotenv

# Load variables from a .env file if one is present in the working directory.
# This is a no-op if the file does not exist, which is what we want in Docker.
load_dotenv()

# Tool version. Bumped when behavior changes in a user-visible way.
TOOL_VERSION = "0.1.0"

# Rubric version. Bumped when the rubric prose or decision principles change.
RUBRIC_VERSION = "mvp-modality-text-v1"

# Default Anthropic model. Can be overridden via the ANTHROPIC_MODEL env var.
DEFAULT_MODEL = "claude-sonnet-4-5"

# Temperature for the LLM call. Set to zero so the same input produces the
# same output as much as possible.
LLM_TEMPERATURE = 0.0

# Maximum tokens for the LLM response. The recommendations array is small
# but the rationales can run several sentences each.
LLM_MAX_TOKENS = 4096


def get_api_key() -> str:
    """Returns the Anthropic API key from the environment.

    Raises a RuntimeError with a clear message if the variable is missing.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or export it."
        )
    return key


def get_model() -> str:
    """Returns the model name from the environment, falling back to the default."""
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
