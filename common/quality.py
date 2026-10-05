"""
Quality metrics for the pipeline outputs.

These are distinct from common.metrics (which measures *operational* cost --
latency, tokens). The functions here measure *quality* and are computed
post-hoc from the agent outputs (profile, ranked results, cover letter). Each
one targets a specific place this system can fail:

  - profile_completeness   -> did extraction produce a usable profile?
  - ranking_quality        -> is the ranking actually discriminating, and do
                              the embedding and LLM methods agree on ORDER?
  - cover_letter_checks    -> is the letter first-person and within length?

They are pure (no LLM calls, no side effects), so they are cheap enough to
print after every run. The heavier LLM-as-judge checks (groundedness,
faithfulness) live in scripts/evaluate_langsmith.py.
"""

from __future__ import annotations

import re
import statistics

# Fields we expect a good extraction to populate. `name`/`years_experience`
# are scalar; the rest are lists. Weighted equally for a simple 0-1 score.
_PROFILE_FIELDS = [
    "name",
    "summary",
    "skills",
    "years_experience",
    "past_titles",
    "education",
    "certifications",
    "languages",
]


def profile_completeness(profile: dict) -> dict:
    """Fraction of expected profile fields that are populated, plus flag count.
    A low score signals a weak parse (e.g. a scanned PDF or an odd layout)."""
    filled = 0
    for field in _PROFILE_FIELDS:
        value = profile.get(field)
        if isinstance(value, list):
            filled += 1 if value else 0
        elif field == "years_experience":
            filled += 1 if value is not None else 0  # 0.0 years still counts as extracted
        else:
            filled += 1 if value else 0
    return {
        "completeness": round(filled / len(_PROFILE_FIELDS), 3),
        "fields_filled": filled,
        "fields_total": len(_PROFILE_FIELDS),
        "flags": len(profile.get("flags", [])),
    }


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation = Pearson on the ranks. Measures whether two
    scorings order items the same way (the right question for a ranking, where
    Pearson on raw scores is misleading)."""
    n = len(a)
    if n < 3:
        return None

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: xs[i])
        rk = [0.0] * n
        i = 0
        while i < n:  # average ranks for ties
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    ra, rb = ranks(a), ranks(b)
    if statistics.pstdev(ra) == 0 or statistics.pstdev(rb) == 0:
        return None
    return round(statistics.correlation(ra, rb), 3)


def ranking_quality(ranked: list[dict]) -> dict:
    """Separation (is the ranking discriminating?) and embedding-vs-LLM rank
    agreement (do the two scoring methods agree on ORDER?)."""
    combined = [r["combined_score"] for r in ranked]
    sims = [r["similarity_score"] for r in ranked]
    llms = [r["llm_score"] / 100 for r in ranked]

    top = max(combined) if combined else 0.0
    median = statistics.median(combined) if combined else 0.0
    return {
        "count": len(ranked),
        "combined_mean": round(statistics.mean(combined), 3) if combined else 0.0,
        "combined_stdev": round(statistics.pstdev(combined), 3) if len(combined) > 1 else 0.0,
        "top_minus_median_gap": round(top - median, 3),
        "spearman_embedding_vs_llm": _spearman(sims, llms),
    }


_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'll|my|me|myself)\b")
# 3rd-person self-reference like "Jane Doe is excited" / "she is applying"
_THIRD_PERSON_HINT = re.compile(r"\b(he|she)\s+(is|has|was|will|brings|holds)\b", re.IGNORECASE)


def cover_letter_checks(letter_text: str, candidate_name: str | None) -> dict:
    """Cheap heuristics guarding the cover-letter format: first-person voice,
    word count, and no obvious third-person self-reference."""
    words = len(letter_text.split())
    first_person_hits = len(_FIRST_PERSON.findall(letter_text))
    third_person = bool(_THIRD_PERSON_HINT.search(letter_text))
    # "<Full Name> is/has/brings ..." -> candidate written about in 3rd person
    name_third_person = False
    if candidate_name:
        name_third_person = bool(
            re.search(rf"{re.escape(candidate_name)}\s+(is|has|was|brings|holds)\b", letter_text)
        )
    return {
        "word_count": words,
        "within_length": 200 <= words <= 500,
        "first_person_hits": first_person_hits,
        "is_first_person": first_person_hits >= 3 and not name_third_person,
        "third_person_self_reference": third_person or name_third_person,
    }
