"""
Single place that loads `.env` into the process environment.

Import this once, as early as possible, in every entry point (Streamlit app,
CLI, standalone test scripts, the MCP server). Child processes spawned by
this one (e.g. the MCP server subprocess) inherit `os.environ` automatically,
so they do not need to load `.env` themselves as long as the parent already
did.
"""

from dotenv import load_dotenv

load_dotenv()
