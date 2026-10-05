"""
Manual smoke test for the Profile Extraction Agent (build order step 3).

Usage:
    python scripts/test_profile_extraction.py [path/to/resume.pdf]

Defaults to sample_data/sample_resume.pdf -- run
sample_data/generate_sample_resume.py first if it doesn't exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import common.config  # noqa: E402,F401  (loads .env)

from agents.profile_extraction import extract_profile  # noqa: E402


def main() -> None:
    default_path = Path(__file__).parent.parent / "sample_data" / "sample_resume.pdf"
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not pdf_path.exists():
        print(f"No PDF found at {pdf_path}. Run sample_data/generate_sample_resume.py first, "
              f"or pass a path to a resume PDF.", file=sys.stderr)
        sys.exit(1)

    profile = extract_profile(pdf_path.read_bytes())
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
