"""
CLI fallback presentation layer -- pretty-prints Phase A results as JSON and
prompts for a job selection to run Phase B. Useful when Streamlit isn't
available (e.g. a notebook or a headless demo).

Usage:
    python -m ui.cli --resume sample_data/sample_resume.pdf --title "Data Analyst"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import common.config  # noqa: E402,F401  (loads .env)

from orchestrator.graph import new_thread_id, run_phase_a, run_phase_b  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Job Application Agent (CLI fallback)")
    parser.add_argument("--resume", required=True, help="Path to a resume file (PDF or image)")
    parser.add_argument("--title", required=True, help="Desired job title")
    parser.add_argument("--location", default="", help="Optional location filter")
    args = parser.parse_args()

    resume_path = Path(args.resume)
    pdf_bytes = resume_path.read_bytes()
    thread_id = new_thread_id()

    print("Running Phase A...", file=sys.stderr)
    result = asyncio.run(
        run_phase_a(pdf_bytes, args.title, args.location, thread_id, resume_filename=resume_path.name)
    )

    if result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({"profile": result.get("profile"), "ranked_results": result.get("ranked_results")}, indent=2))

    ranked = result.get("ranked_results") or []
    if not ranked:
        return

    print("\nAvailable job IDs:", [r["job"]["id"] for r in ranked], file=sys.stderr)
    selected_id = input("Select a job_id for the cover letter (or press Enter to skip): ").strip()
    if not selected_id:
        return

    phase_b_result = asyncio.run(run_phase_b(thread_id, selected_id))
    if phase_b_result.get("error"):
        print(f"Error: {phase_b_result['error']}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(phase_b_result.get("cover_letter"), indent=2))


if __name__ == "__main__":
    main()
