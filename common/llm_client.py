"""
Thin wrapper around the Groq chat-completions API.

Centralizes, for every agent:
  - API-key fallback: GROQ_API_KEYS may hold a comma-separated list of keys;
    when a call fails with a rate-limit/auth/server error, the client rotates
    to the next key and retries, so one exhausted free-tier key doesn't kill
    the demo. GROQ_API_KEY (single key) is still supported.
  - vision calls (image resumes are transcribed by a multimodal model)
  - token-usage + latency metrics (fed into common.metrics, printed to the
    console at the end of each orchestrator phase)
  - LangSmith tracing spans for every LLM call (no-op if langsmith is not
    configured; enable with LANGSMITH_TRACING=true + LANGSMITH_API_KEY)
"""

from __future__ import annotations

import json
import os
import time

from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError

from common import metrics

try:  # optional: tracing is a no-op decorator when langsmith isn't installed/configured
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*d_args, **d_kwargs):
        def wrap(fn):
            return fn
        return wrap if not (d_args and callable(d_args[0])) else d_args[0]

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

_clients: dict[str, Groq] = {}
_key_index = 0


def _get_keys() -> list[str]:
    """GROQ_API_KEYS (comma-separated list, for fallback) or GROQ_API_KEY."""
    multi = os.environ.get("GROQ_API_KEYS", "")
    keys = [k.strip() for k in multi.replace(";", ",").split(",") if k.strip()]
    single = os.environ.get("GROQ_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    if not keys:
        raise RuntimeError(
            "No Groq API key configured. Set GROQ_API_KEY (or GROQ_API_KEYS=key1,key2,... "
            "for fallback) in .env -- free keys at https://console.groq.com/keys."
        )
    return keys


def _client_for(key: str) -> Groq:
    if key not in _clients:
        _clients[key] = Groq(api_key=key)
    return _clients[key]


def _create_with_fallback(model: str, **kwargs):
    """Run one chat.completions.create, rotating across configured API keys on
    rate-limit/auth/connection/server errors. Sticky: once a key works, later
    calls keep starting from it (_key_index survives between calls)."""
    global _key_index
    keys = _get_keys()
    last_error: Exception | None = None
    for attempt in range(len(keys)):
        key = keys[(_key_index + attempt) % len(keys)]
        try:
            started = time.perf_counter()
            response = _client_for(key).chat.completions.create(model=model, **kwargs)
            elapsed = time.perf_counter() - started
            usage = getattr(response, "usage", None)
            metrics.registry.record_llm_call(
                model,
                elapsed,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
            _key_index = (_key_index + attempt) % len(keys)
            return response
        except (RateLimitError, AuthenticationError, APIConnectionError) as e:
            last_error = e
            metrics.registry.record_key_rotation(model)
            print(f"[llm_client] key #{(_key_index + attempt) % len(keys) + 1} failed "
                  f"({type(e).__name__}), trying next key..." if attempt + 1 < len(keys)
                  else f"[llm_client] key #{(_key_index + attempt) % len(keys) + 1} failed ({type(e).__name__})")
        except APIStatusError as e:
            if e.status_code >= 500:  # server-side issue: worth retrying on another key/deployment
                last_error = e
                metrics.registry.record_key_rotation(model)
            else:  # 4xx other than auth/ratelimit is our bug -- don't mask it by rotating
                raise
    raise RuntimeError(f"All configured Groq API keys failed. Last error: {last_error}") from last_error


@traceable(run_type="llm", name="groq.chat_json")
def chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
    """Call the LLM and parse its reply as a JSON object (Groq JSON mode)."""
    response = _create_with_fallback(
        MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}") from e


@traceable(run_type="llm", name="groq.chat_text")
def chat_text(system_prompt: str, user_prompt: str, max_tokens: int = 900, temperature: float = 0.4) -> str:
    """Call the LLM and return its free-text reply (used for cover letters)."""
    response = _create_with_fallback(
        MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


@traceable(run_type="llm", name="groq.chat_vision")
def chat_vision(instruction: str, image_data_url: str, max_tokens: int = 3000) -> str:
    """Send an image (as a base64 data URL) plus an instruction to the
    multimodal model. Used to transcribe image resumes into plain text."""
    response = _create_with_fallback(
        VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
