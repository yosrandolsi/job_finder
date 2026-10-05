"""
MCP server exposing one tool: search_jobs(query, location, max_results).

Providers: Adzuna (primary, needs ADZUNA_APP_ID/ADZUNA_APP_KEY) with
Arbeitnow as a no-key fallback if Adzuna is unconfigured, quota-limited, or
errors.

Security notes
---------------
- Transport is stdio only: this server is spawned as a local subprocess by
  the Job Search Agent (see mcp_server/client.py) and never listens on a
  network socket, so it has no remote attack surface for this POC.
- All logging goes to stderr (see logging.basicConfig below). Never print to
  stdout in this process -- stdout is the JSON-RPC channel for the stdio
  transport, and any stray print() would corrupt the protocol stream.
- Inputs (query/location) are cleaned with sanitize_single_line and
  length-capped before use, and are always passed to `requests` via
  `params=` (never string-concatenated into a URL), which rules out
  header/URL injection into the upstream APIs.
- Outbound hosts are fixed constants, not derived from user input. The
  Adzuna country code (from an env var, not user input) is still validated
  against an allowlist before being placed in the URL path, since it is the
  one piece of "external" data that ends up in the path rather than a query
  param.
- Credentials are read from environment variables only and are never
  included in tool output, logs, or error messages returned to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from mcp.server.fastmcp import FastMCP

import common.config  # noqa: F401  (loads .env if this server is ever run standalone)
from common.security import MAX_LOCATION_LEN, MAX_QUERY_LEN, MAX_RESULTS_CAP, sanitize_single_line
from schemas.models import JobPosting

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("job_search_mcp")

ADZUNA_HOST = "api.adzuna.com"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
ALLOWED_COUNTRIES = {"us", "gb", "de", "fr", "ca", "au", "nl", "es", "it", "pl", "at", "ch", "in", "sg"}
REQUEST_TIMEOUT = 10

mcp = FastMCP("job-search")


class ProviderError(Exception):
    """Raised when a job-search provider can't be used or fails."""


def _adzuna_request(query: str, location: str, max_results: int, use_what_or: bool) -> list[JobPosting]:
    """One Adzuna API call. `use_what_or` switches from all-words to any-word matching."""
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise ProviderError("Adzuna credentials are not configured")

    country = os.environ.get("ADZUNA_COUNTRY", "us").lower()
    if country not in ALLOWED_COUNTRIES:
        log.warning("Unrecognized ADZUNA_COUNTRY '%s', defaulting to 'us'", country)
        country = "us"

    url = f"https://{ADZUNA_HOST}/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": max_results,
        "content-type": "application/json",
    }
    params["what_or" if use_what_or else "what"] = query
    if location:
        params["where"] = location

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError("Adzuna request failed") from e

    data = resp.json()
    postings: list[JobPosting] = []
    for item in data.get("results", [])[:max_results]:
        postings.append(
            JobPosting(
                id=str(item.get("id", "")),
                title=item.get("title", "Unknown"),
                company=(item.get("company") or {}).get("display_name", "Unknown"),
                description=item.get("description", ""),
                location=(item.get("location") or {}).get("display_name", location or "Unknown"),
                url=item.get("redirect_url", ""),
                source="adzuna",
            )
        )
    return postings


def _search_adzuna(query: str, location: str, max_results: int) -> list[JobPosting]:
    """Adzuna with a progressive-relaxation retry chain, so a too-narrow query
    (e.g. a location outside the configured country, or an uncommon title)
    degrades to broader results instead of returning nothing:
      1. all words + location
      2. all words, no location (location may not exist in this country index)
      3. any word, no location (title phrasing may not match posting titles)
    """
    attempts = [
        (query, location, False),
        (query, "", False),
        (query, "", True),
    ]
    seen_attempts = set()
    for q, loc, what_or in attempts:
        key = (q, loc, what_or)
        if key in seen_attempts:
            continue  # skip duplicate attempts (e.g. when location was empty to begin with)
        seen_attempts.add(key)
        postings = _adzuna_request(q, loc, max_results, what_or)
        if postings:
            if (loc, what_or) != (location, False):
                log.info("Adzuna: query relaxed (location=%r, what_or=%s) to find results", loc, what_or)
            return postings
    return []


def _search_arbeitnow(query: str, location: str, max_results: int) -> list[JobPosting]:
    try:
        resp = requests.get(ARBEITNOW_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError("Arbeitnow request failed") from e

    data = resp.json()
    # Token-overlap scoring instead of exact-phrase matching: "data analyst"
    # should still match a "Senior Data Analyst (m/f/d)" posting, and a query
    # with no token hits at all degrades to the newest postings rather than
    # an empty list (the Matching Agent downstream will rank fit anyway).
    tokens = [t for t in query.lower().split() if len(t) > 2]
    loc = location.lower()

    scored: list[tuple[int, dict]] = []
    for item in data.get("data", []):
        title = (item.get("title") or "").lower()
        description = (item.get("description") or "").lower()
        score = sum(3 for t in tokens if t in title) + sum(1 for t in tokens if t in description)
        job_location = (item.get("location") or "").lower()
        if loc and (loc in job_location or item.get("remote")):
            score += 2  # location match is a bonus, never a hard filter
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Prefer postings with at least one token hit; pad with newest otherwise.
    relevant = [item for score, item in scored if score > 0] or [item for _, item in scored]

    postings: list[JobPosting] = []
    for item in relevant[:max_results]:
        postings.append(
            JobPosting(
                id=str(item.get("slug", len(postings))),
                title=item.get("title", ""),
                company=item.get("company_name", "Unknown"),
                description=item.get("description", ""),
                location=item.get("location", "") or ("Remote" if item.get("remote") else "Unknown"),
                url=item.get("url", ""),
                source="arbeitnow",
            )
        )
    return postings


@mcp.tool()
def search_jobs(query: str, location: str = "", max_results: int = 15) -> str:
    """Search job postings by title/keywords and (optional) location.

    Returns a JSON array of job posting objects (id, title, company,
    description, location, url, source). Tries Adzuna first and falls back
    to Arbeitnow if Adzuna is unavailable, unconfigured, or errors.
    """
    clean_query = sanitize_single_line(query, MAX_QUERY_LEN)
    clean_location = sanitize_single_line(location, MAX_LOCATION_LEN)
    capped_results = max(1, min(int(max_results), MAX_RESULTS_CAP))

    if not clean_query:
        return json.dumps([])

    postings: list[JobPosting] = []
    try:
        postings = _search_adzuna(clean_query, clean_location, capped_results)
    except ProviderError as e:
        log.info("Adzuna unavailable (%s), falling back to Arbeitnow", e)

    if not postings:
        try:
            postings = _search_arbeitnow(clean_query, clean_location, capped_results)
        except ProviderError as e:
            log.warning("Arbeitnow fallback also failed: %s", e)
            postings = []

    return json.dumps([p.model_dump() for p in postings])


if __name__ == "__main__":
    mcp.run(transport="stdio")
