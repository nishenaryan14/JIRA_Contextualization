#!/usr/bin/env python
"""
Jira Contextualization Engine — Entry Point.

Runs the complete 6-stage knowledge build pipeline:
  1. Ingest & Normalize CSV
  2. Extract Requirements (DeepSeek LLM)
  3. Build Relationships
  4. Validate Knowledge (Gemini LLM)
  5. Publish Artifacts
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def run() -> None:
    """Run the Jira Contextualization pipeline."""
    # Load environment
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    # Determine CSV path
    csv_path = os.getenv("CSV_INPUT_PATH", "")
    if not Path(csv_path).is_absolute():
        csv_path = str(Path(__file__).parent.parent.parent / csv_path)

    if not Path(csv_path).exists():
        # Try to find any CSV in data/raw/
        raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        csv_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
        if csv_files:
            csv_path = str(csv_files[0])
        else:
            print("❌ No CSV file found. Place your Jira export in data/raw/")
            print(f"   Or set CSV_INPUT_PATH in .env")
            sys.exit(1)

    output_dir = os.getenv("OUTPUT_DIR", "output")
    if not Path(output_dir).is_absolute():
        output_dir = str(Path(__file__).parent.parent.parent / output_dir)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  JIRA CONTEXTUALIZATION ENGINE v1.0".center(68) + "║")
    print("║" + "  From Raw Jira Data to Validated Knowledge Base".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  📂 Input:  {csv_path}")
    print(f"  📁 Output: {output_dir}")
    print(f"  🤖 Extraction LLM: DeepSeek Chat")
    print(f"  🤖 Validation LLM: Gemini 2.0 Flash")

    # Run the flow
    from jira_contextualization.flow import JiraContextualizationFlow

    flow = JiraContextualizationFlow()
    flow.state.csv_path = csv_path
    flow.state.output_dir = output_dir

    result = flow.kickoff()
    print(f"\n  Pipeline result: {result}")


if __name__ == "__main__":
    run()
