#!/usr/bin/env python3
"""Run DevAgent with actual Gemma 4 inference (requires a compatible GPU)."""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import DevAgent
from agent.gemma4_backend import Gemma4Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 DevAgent with real model inference")
    parser.add_argument("--model_path", required=True, help="Local Kaggle model directory or Hugging Face model ID")
    parser.add_argument("--repo_path", required=True, help="Repository to inspect and modify")
    parser.add_argument("--issue", required=True, help="Software issue to fix")
    parser.add_argument("--max_turns", type=int, default=24)
    parser.add_argument("--max_tool_calls", type=int, default=30)
    parser.add_argument("--time_budget_sec", type=int, default=600)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--adapter_path", help="Optional SFT/GRPO LoRA adapter directory")
    parser.add_argument("--trajectory_path", help="Optional JSON trajectory output path")
    args = parser.parse_args()

    repo = Path(args.repo_path).resolve()
    if not repo.is_dir():
        raise SystemExit(f"Repository directory does not exist: {repo}")

    model = Gemma4Generator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        adapter_path=args.adapter_path,
    )
    agent = DevAgent(
        repo_path=str(repo),
        issue=args.issue,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        time_budget_sec=args.time_budget_sec,
        trajectory_path=args.trajectory_path,
    )
    result = agent.run(model)
    print("\n[FINAL AGENT RESULT]")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
