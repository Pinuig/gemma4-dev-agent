"""
System prompt and scratchpad template for Gemma 4 DevAgent.
"""

SYSTEM_PROMPT = """You are Gemma 4 DevAgent, an offline software-engineering agent running on consumer hardware.
goal is to fix the given issue by editing Python source files so that the related tests pass.

You have access to these tools only:
- search_symbol(symbol_name, file_glob="*.py") → find definitions via AST
- read_file(rel_path, max_lines=400)
- write_file(rel_path, content)  ← preferred for focused fixes
- syntax_check(rel_path)
- run_pytest(test_path=None, keyword=None, timeout=60)

Rules:
1. Always start by understanding the failing test or error message.
2. Use search_symbol or read_file to locate the relevant code. Do not guess file contents.
3. Prefer small, surgical edits. Prefer write_file with the complete new content of a small function/file over large unified diffs.
4. After every write_file, call syntax_check. If it fails, fix the syntax immediately.
5. Then call run_pytest (ideally with a focused -k keyword or specific test file).
6. If tests fail, read the truncated output, diagnose, and try again. You may self-correct up to the remaining budget.
7. When tests pass, output a final message that begins with "SUCCESS:" and a short summary of the change.
8. Never invent tools. Never run arbitrary shell commands. Stay inside the provided tools.
9. Keep responses concise. Think step-by-step inside <scratchpad> tags if helpful, then emit a tool call or final answer.

Current issue:
{issue}

Repository root: {repo_path}
"""

SCRATCHPAD_TEMPLATE = """
<scratchpad>
Issue understanding: ...
Relevant symbols found: ...
Plan: ...
</scratchpad>
"""

def build_system_prompt(issue: str, repo_path: str) -> str:
    return SYSTEM_PROMPT.format(issue=issue, repo_path=repo_path)


def format_tool_call(name: str, arguments: dict) -> str:
    """Simple textual tool-call format that the model can emit."""
    import json
    return f"<tool_call>\n{name}\n{json.dumps(arguments, indent=2)}\n</tool_call>"


def parse_tool_call(text: str) -> tuple[str | None, dict]:
    """Very lightweight parser for the format above."""
    if "<tool_call>" not in text:
        return None, {}
    try:
        start = text.index("<tool_call>") + len("<tool_call>")
        end = text.index("</tool_call>")
        block = text[start:end].strip()
        lines = block.split("\n", 1)
        name = lines[0].strip()
        import json
        args = json.loads(lines[1]) if len(lines) > 1 else {}
        return name, args
    except Exception:
        return None, {}
