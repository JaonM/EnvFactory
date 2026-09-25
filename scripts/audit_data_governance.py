#!/usr/bin/env python3
"""Audit a sandbox's synthetic-data declaration and external-model payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from env_factory.data_governance import audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env")
    agent_model = os.getenv("LLM_MODEL", "")
    runtime_model = os.getenv("SANDBOX_LLM_MODEL") or agent_model
    report = audit(
        args.root.resolve(),
        agent_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        agent_model=agent_model,
        runtime_base_url=(
            os.getenv("SANDBOX_LLM_BASE_URL")
            or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        ),
        runtime_model=runtime_model,
    )
    output = args.output or args.root / "data_governance.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["eligible_for_external_model_processing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
