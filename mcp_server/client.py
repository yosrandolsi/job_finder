"""
MCP client used by the Job Search Agent to call the `search_jobs` tool.

Spawns mcp_server/job_search_server.py as a local stdio subprocess per call.
This is the actual MCP protocol round-trip (JSON-RPC over stdio), not a
direct function import, so the Job Search Agent and the job-search tool are
genuinely decoupled processes communicating over MCP.
"""

from __future__ import annotations

import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.job_search_server"],
    cwd=_PROJECT_ROOT,
)


async def call_search_jobs(query: str, location: str = "", max_results: int = 15) -> list[dict]:
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_jobs",
                arguments={"query": query, "location": location, "max_results": max_results},
            )
            if result.isError:
                raise RuntimeError(f"search_jobs tool returned an error: {result.content}")

            text = ""
            for block in result.content:
                if getattr(block, "text", None):
                    text = block.text
                    break
            if not text:
                return []
            return json.loads(text)
