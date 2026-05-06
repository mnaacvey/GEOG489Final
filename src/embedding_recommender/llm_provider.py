"""Sends the prompt to the Anthropic API and returns validated recommendations."""

import json
import anthropic
from jsonschema import ValidationError, validate
from . import config
from .schema import RECOMMENDATIONS_SCHEMA, validate_recommendations_business_rules
from .timing import timed

SYSTEM_PROMPT = (
    "You are an embedding strategy advisor. Apply the rubric in the user "
    "message to the dataset profile and intent. Return only a valid JSON "
    "array matching the schema in the rubric. No text before or after."
)


@timed
def generate_recommendations(profile, intent, rubric_text, api_key, model, temperature=0.0):
    """Calls the Anthropic API once and returns validated recommendations.

    Raises RuntimeError if the response is not valid JSON or fails the
    schema or business-rule checks. Single attempt; the operator reruns
    on failure.
    """
    client = anthropic.Anthropic(api_key=api_key)
    user_message = (
        f"RUBRIC:\n{rubric_text}\n\n"
        f"DATASET PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
        f"USER INTENT:\n{json.dumps(intent, indent=2)}\n\n"
        "Return only the JSON array of five recommendations."
    )

    response = client.messages.create(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    cleaned = _strip_code_fences(raw)

    try:
        recs = json.loads(cleaned)
        validate(instance=recs, schema=RECOMMENDATIONS_SCHEMA)
        validate_recommendations_business_rules(recs)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise RuntimeError(
            f"LLM returned an invalid response: {e}\nResponse was: {raw}"
        ) from e

    return recs


def _strip_code_fences(text: str) -> str:
    """Removes leading/trailing markdown code fences if present."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
