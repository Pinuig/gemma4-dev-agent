"""
Main agent execution loop for Gemma 4 DevAgent.
This is a reference implementation that can be driven by a local model
or by the competition harness (Google ADK style).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import call_tool, AVAILABLE_TOOLS
from .prompt import build_system_prompt, parse_tool_call


class DevAgent:
    def __init__(
        self,
        repo_path: str,
        issue: str,
        max_turns: int = 30,
        max_tool_calls: int = 40,
        time_budget_sec: int = 480,  # 8 minutes default
    ):
        self.repo_path = str(Path(repo_path).resolve())
        self.issue = issue
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.time_budget_sec = time_budget_sec
        self.history: List[Dict[str, str]] = []
        self.tool_calls = 0
        self.start_time = time.time()

    def remaining_time(self) -> float:
        return max(0.0, self.time_budget_sec - (time.time() - self.start_time))

    def run(self, model_generate_fn) -> Dict[str, Any]:
        """
        model_generate_fn(messages: List[Dict]) -> str
        messages follow OpenAI-style role/content.
        """
        system = build_system_prompt(self.issue, self.repo_path)
        messages = [{"role": "system", "content": system}]

        for turn in range(self.max_turns):
            if self.tool_calls >= self.max_tool_calls or self.remaining_time() < 10:
                return self._finish(False, "Budget exhausted")

            # Ask the model
            response = model_generate_fn(messages)
            messages.append({"role": "assistant", "content": response})

            # Check for success
            if response.strip().startswith("SUCCESS:"):
                return self._finish(True, response)

            # Try to parse a tool call
            tool_name, tool_args = parse_tool_call(response)
            if tool_name is None:
                # Model did not emit a tool call – give a gentle reminder
                messages.append({
                    "role": "user",
                    "content": "Please either call a tool using the <tool_call> format or finish with SUCCESS: if the tests already pass."
                })
                continue

            # Execute the tool
            self.tool_calls += 1
            try:
                result = call_tool(tool_name, self.repo_path, **tool_args)
            except Exception as e:
                result = {"error": str(e)}

            # Feed observation back
            observation = json.dumps(result, indent=2, default=str)[:4000]
            messages.append({
                "role": "user",
                "content": f"Tool result ({tool_name}):\n{observation}"
            })

        return self._finish(False, "Max turns reached")

    def _finish(self, success: bool, message: str) -> Dict[str, Any]:
        return {
            "success": success,
            "message": message,
            "tool_calls": self.tool_calls,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "history_len": len(self.history),
        }


def dummy_generate(messages: List[Dict]) -> str:
    """Placeholder for local testing without a real model."""
    last = messages[-1]["content"] if messages else ""
    if "Tool result" in last and "PASSED" in last:
        return "SUCCESS: Tests are green after the last edit."
    # Fake a search
    return """<tool_call>
search_symbol
{"symbol_name": "add"}
</tool_call>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--max_turns", type=int, default=20)
    args = parser.parse_args()

    agent = DevAgent(args.repo_path, args.issue, max_turns=args.max_turns)
    result = agent.run(dummy_generate)
    print(json.dumps(result, indent=2))
