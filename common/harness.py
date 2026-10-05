"""
Agent harness: a uniform execution wrapper that every agent and subagent
runs through, providing hooks, metrics, and (optional) LangSmith spans.

Concepts, mapped to this project:
  - HARNESS  = `run_agent` / `run_agent_sync`: the controlled entry point
    that executes an agent function, times it, records metrics, and fires
    lifecycle hooks. The orchestrator never calls an agent directly -- it
    always goes through the harness, so cross-cutting behavior (logging,
    metrics, guardrails) lives in exactly one place.
  - HOOKS    = registered callbacks fired at `before_agent`, `after_agent`,
    and `on_error`. Default hooks do console logging + metrics recording;
    extra hooks can be registered without touching any agent code
    (e.g. a hook that blocks an agent run if input exceeds a size budget).
  - SUBAGENT = a smaller, single-purpose LLM helper invoked *inside* a main
    agent (e.g. the QueryNormalizerSubagent inside the Job Search Agent, or
    the MatchScoringSubagent the Matching Agent runs once per job posting).
    Subagents run through the same harness with quiet=True so they are
    metered/traceable without flooding the console.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from common import metrics

try:  # optional LangSmith tracing (no-op when not installed/configured)
    from langsmith import trace as _ls_trace
except ImportError:  # pragma: no cover
    _ls_trace = None

Hook = Callable[[str, dict], None]

_hooks: dict[str, list[Hook]] = {"before_agent": [], "after_agent": [], "on_error": []}


def register_hook(event: str, hook: Hook) -> None:
    """Register a callback for 'before_agent' | 'after_agent' | 'on_error'.
    Each hook receives (agent_name, info_dict)."""
    if event not in _hooks:
        raise ValueError(f"Unknown hook event '{event}'. Valid: {list(_hooks)}")
    _hooks[event].append(hook)


def _fire(event: str, agent_name: str, info: dict) -> None:
    for hook in _hooks[event]:
        try:
            hook(agent_name, info)
        except Exception as e:  # a broken hook must never take down the pipeline
            print(f"[harness] hook {hook.__name__} failed on {event}: {e}")


# --- default hooks: console logging (skipped for quiet subagent runs) --------

# ASCII markers only: the Windows console default code page (cp1252) cannot
# encode symbols like a check mark, and a logging hook must never crash.
def _log_start(agent_name: str, info: dict) -> None:
    if not info.get("quiet"):
        print(f"[harness] >> {agent_name} started")


def _log_end(agent_name: str, info: dict) -> None:
    if not info.get("quiet"):
        print(f"[harness] OK {agent_name} finished in {info['seconds']:.2f}s")


def _log_error(agent_name: str, info: dict) -> None:
    print(f"[harness] !! {agent_name} failed after {info['seconds']:.2f}s: {info['error']}")


register_hook("before_agent", _log_start)
register_hook("after_agent", _log_end)
register_hook("on_error", _log_error)


# --- execution wrappers -------------------------------------------------------

async def run_agent(name: str, fn: Callable[..., Awaitable[Any]], *args, quiet: bool = False, **kwargs) -> Any:
    """Run an async agent function through the harness."""
    _fire("before_agent", name, {"quiet": quiet})
    started = time.perf_counter()
    try:
        if _ls_trace is not None:
            with _ls_trace(name=name, run_type="chain"):
                result = await fn(*args, **kwargs)
        else:
            result = await fn(*args, **kwargs)
    except Exception as e:
        seconds = time.perf_counter() - started
        metrics.registry.record_agent(name, seconds, error=True)
        _fire("on_error", name, {"quiet": quiet, "seconds": seconds, "error": e})
        raise
    seconds = time.perf_counter() - started
    metrics.registry.record_agent(name, seconds)
    _fire("after_agent", name, {"quiet": quiet, "seconds": seconds})
    return result


def run_agent_sync(name: str, fn: Callable[..., Any], *args, quiet: bool = False, **kwargs) -> Any:
    """Run a sync agent/subagent function through the harness."""
    _fire("before_agent", name, {"quiet": quiet})
    started = time.perf_counter()
    try:
        if _ls_trace is not None:
            with _ls_trace(name=name, run_type="chain"):
                result = fn(*args, **kwargs)
        else:
            result = fn(*args, **kwargs)
    except Exception as e:
        seconds = time.perf_counter() - started
        metrics.registry.record_agent(name, seconds, error=True)
        _fire("on_error", name, {"quiet": quiet, "seconds": seconds, "error": e})
        raise
    seconds = time.perf_counter() - started
    metrics.registry.record_agent(name, seconds)
    _fire("after_agent", name, {"quiet": quiet, "seconds": seconds})
    return result
