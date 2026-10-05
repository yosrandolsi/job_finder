"""
Manual smoke test for the MCP job-search server: confirms the MCP
client/server round-trip works end to end (build order step 2).

Usage:
    python scripts/test_job_search.py "data analyst" "Paris"

With no Adzuna credentials configured, this also exercises the Arbeitnow
fallback path.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import common.config  # noqa: E402,F401  (loads .env)

from mcp_server.client import call_search_jobs  # noqa: E402


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "data analyst"
    location = sys.argv[2] if len(sys.argv) > 2 else ""
    results = await call_search_jobs(query=query, location=location, max_results=5)
    print(json.dumps(results, indent=2))
    print(f"\n{len(results)} postings found.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
