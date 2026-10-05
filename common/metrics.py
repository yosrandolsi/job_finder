"""
In-process performance metrics for the agent pipeline.

Collects two kinds of measurements:
  - per-agent timings (recorded by the harness around every agent/subagent run)
  - per-LLM-call stats (model, latency, token usage -- recorded by llm_client)

`print_summary()` renders a console table at the end of each orchestrator
phase. This is deliberately console-only (not shown in the UI), matching the
evaluation requirement; deeper offline evaluation lives in
scripts/evaluate_phoenix.py.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _AgentStats:
    calls: int = 0
    errors: int = 0
    total_seconds: float = 0.0


@dataclass
class _LLMStats:
    calls: int = 0
    total_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    key_rotations: int = 0  # how many times we fell back to another API key


@dataclass
class MetricsRegistry:
    agents: dict[str, _AgentStats] = field(default_factory=lambda: defaultdict(_AgentStats))
    llm: dict[str, _LLMStats] = field(default_factory=lambda: defaultdict(_LLMStats))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_agent(self, name: str, seconds: float, error: bool = False) -> None:
        with self._lock:
            stats = self.agents[name]
            stats.calls += 1
            stats.total_seconds += seconds
            if error:
                stats.errors += 1

    def record_llm_call(self, model: str, seconds: float, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            stats = self.llm[model]
            stats.calls += 1
            stats.total_seconds += seconds
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens

    def record_key_rotation(self, model: str) -> None:
        with self._lock:
            self.llm[model].key_rotations += 1

    def print_summary(self, title: str = "PIPELINE METRICS") -> None:
        print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")
        if self.agents:
            print(f"  {'Agent':<32} {'calls':>5} {'errors':>6} {'total s':>8} {'avg s':>7}")
            print(f"  {'-' * 62}")
            for name, s in self.agents.items():
                avg = s.total_seconds / s.calls if s.calls else 0.0
                print(f"  {name:<32} {s.calls:>5} {s.errors:>6} {s.total_seconds:>8.2f} {avg:>7.2f}")
        if self.llm:
            print()
            print(f"  {'LLM model':<38} {'calls':>5} {'tok in':>7} {'tok out':>8} {'rotations':>9}")
            print(f"  {'-' * 70}")
            for model, s in self.llm.items():
                print(f"  {model:<38} {s.calls:>5} {s.prompt_tokens:>7} {s.completion_tokens:>8} {s.key_rotations:>9}")
        print(f"{'=' * 72}\n")


# Single shared registry for the process. reset() is available for eval runs
# that want isolated numbers.
registry = MetricsRegistry()


def reset() -> None:
    global registry
    registry = MetricsRegistry()
