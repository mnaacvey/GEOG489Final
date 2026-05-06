"""Loads environment variables and exposes module-level constants."""

import getpass
import os
import sys

from dotenv import load_dotenv

# Load variables from a .env file if one is present in the working directory.
# This is a no-op if the file does not exist, which is what we want in Docker.
load_dotenv()

# Tool version. Bumped when behavior changes in a user-visible way.
TOOL_VERSION = "0.1.1"

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

# Sentinel value shipped in .env.example. Treated the same as a missing key.
_PLACEHOLDER_KEY = "sk-ant-replace-me"


def get_api_key() -> str:
    """Returns the Anthropic API key.

    Looks at the ANTHROPIC_API_KEY env var (loaded from .env if present).
    If the var is missing or holds the .env.example placeholder, prompts
    the operator interactively. Falls back to a clear error when there is
    no terminal attached, so non-interactive callers still fail loudly.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key and key != _PLACEHOLDER_KEY:
        return key

    if not sys.stdin.isatty():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or export it."
        )

    print(
        "ANTHROPIC_API_KEY is not set in the environment or .env file.",
        file=sys.stderr,
    )
    print(
        "Paste your Anthropic API key. Input is hidden.",
        file=sys.stderr,
    )
    print(
        "To skip this prompt on future runs, put the key in .env as "
        "ANTHROPIC_API_KEY=<your-key>.",
        file=sys.stderr,
    )
    entered = getpass.getpass("API key: ").strip()
    if not entered:
        raise RuntimeError("No API key provided.")
    return entered


def get_model() -> str:
    """Returns the model name from the environment, falling back to the default."""
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
