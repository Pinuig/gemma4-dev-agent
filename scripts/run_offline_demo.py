"""
Run a fully offline demonstration of the DevAgent tool loop.

IMPORTANT:
This uses a scripted policy to exercise the real agent loop and tools.
It does NOT call Gemma 4. Use it to validate/record the offline tool pipeline.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import DevAgent
from agent.prompt import format_tool_call
FIXTURE = PROJECT_ROOT / "examples" / "demo_repo"
WORKSPACE = Path("/tmp/gemma4-devagent-demo")


def prepare_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(FIXTURE, WORKSPACE)


class OfflineDemoPolicy:
    """A deterministic policy that exercises a fail-then-repair trajectory."""

    def __init__(self) -> None:
        self.actions = [
            ("search_symbol", {"symbol_name": "add"}),
            ("read_file", {"rel_path": "calculator.py"}),
            # Deliberately imperfect first patch: the test should fail.
            ("write_file", {
                "rel_path": "calculator.py",
                "content": "def add(a, b):\n    return a + b + 1\n",
            }),
            ("syntax_check", {"rel_path": "calculator.py"}),
            ("run_pytest", {}),
            # The next model turn receives the pytest failure as an observation.
            ("read_file", {"rel_path": "checks_demo.py"}),
            # Second patch corrects the implementation.
            ("write_file", {
                "rel_path": "calculator.py",
                "content": "def add(a, b):\n    return a + b\n",
            }),
            ("syntax_check", {"rel_path": "calculator.py"}),
            ("run_pytest", {}),
        ]
        self.index = 0

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        # Print the previous tool observation so the terminal recording shows
        # what the policy receives before choosing its next action.
        if len(messages) > 1 and messages[-1]["role"] == "user":
            observation = messages[-1]["content"]
            if observation.startswith("TOOL_RESULT "):
                print("\n[OBSERVATION]")
                print(observation)

        # Only report success after the final pytest call actually passes.
        if self.index >= len(self.actions):
            last = messages[-1]["content"] if messages else ""
            if '"summary": "PASSED"' in last and '"passed": true' in last:
                return "SUCCESS: Corrected calculator.add; test_add_negative passes."
            return "The demonstration did not reach a passing test."

        name, arguments = self.actions[self.index]
        self.index += 1
        response = format_tool_call(name, arguments)
        print("\n[AGENT TOOL CALL]")
        print(response)
        return response


def main() -> int:
    print("=" * 72)
    print("GEMMA 4 DEVAGENT — OFFLINE TOOL-LOOP DEMONSTRATION")
    print("Policy: scripted demonstration; no Gemma 4 inference is performed.")
    print("=" * 72)
    print("\n[ISSUE]")
    print("The add function returns wrong results for negative numbers.")
    print("Make test_add_negative pass.")

    prepare_workspace()
    print(f"\n[WORKSPACE] {WORKSPACE}")
    print("[INITIAL BUG]")
    print((WORKSPACE / "calculator.py").read_text(encoding="utf-8"))

    agent = DevAgent(
        repo_path=str(WORKSPACE),
        issue="The add function returns wrong results for negative numbers. "
              "Make test_add_negative pass.",
        max_turns=15,
        max_tool_calls=15,
        time_budget_sec=120,
    )
    result = agent.run(OfflineDemoPolicy())

    print("\n[FINAL AGENT RESULT]")
    print(json.dumps(result, indent=2))
    print("\n[FINAL SOURCE]")
    print((WORKSPACE / "calculator.py").read_text(encoding="utf-8"))

    if result.get("success"):
        print("\n" + "=" * 72)
        print("SUCCESS — TESTS GREEN")
        print("=" * 72)
        return 0

    print("\nDEMO FAILED — inspect the output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
