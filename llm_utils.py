"""
Utility functions for LLM providers.
"""

import json
import logging
import re
from typing import Any, Optional
from config import provider_for
from models import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON content from think tags, markdown code blocks, or surrounding prose.

    Args:
        response_text: Text that may contain JSON wrapped in markdown code blocks

    Returns:
        Cleaned string ready for JSON parsing
    """
    if not response_text:
        return ""

    cleaned = _THINK_BLOCK_RE.sub("", str(response_text)).strip()

    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        if cleaned.lower().startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    # Fast path: already valid JSON without surrounding prose.
    if cleaned:
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        # Recover the largest valid top-level JSON object/array wrapped in prose.
        decoder = json.JSONDecoder()
        best_slice: Optional[str] = None
        pos = 0
        length = len(cleaned)
        while pos < length:
            ch = cleaned[pos]
            if ch in ("{", "["):
                try:
                    _, end_offset = decoder.raw_decode(cleaned[pos:])
                    candidate = cleaned[pos : pos + end_offset].strip()
                    if best_slice is None or len(candidate) > len(best_slice):
                        best_slice = candidate
                    pos += max(1, end_offset)
                    continue
                except json.JSONDecodeError:
                    pass
            pos += 1

        if best_slice is not None:
            return best_slice

    return cleaned


def parse_json_response(response_text: str) -> Any:
    """Extract and parse JSON from an LLM response string."""
    cleaned = extract_json_from_response(response_text)
    if not cleaned:
        raise ValueError("LLM response contained no JSON content")
    return json.loads(cleaned)


def initialize_llm_provider(model_name: str) -> Any:
    """
    Initialize an OpenAI-compatible LLM provider for the given model,
    resolving base_url / api_key / structured-output mode from providers.json.
    """
    cfg = provider_for(model_name)
    logger.info(f"🔄 Using model {model_name} via {cfg['base_url']}")
    return OpenAICompatibleProvider(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        structured_output=cfg["structured_output"],
        extra_body=cfg["extra_body"],
    )
