"""Basic unit tests for the agent tools."""

import tempfile
from pathlib import Path

from agent.tools import search_symbol, syntax_check, write_file, read_file


def test_search_symbol_finds_function():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "math_utils.py").write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"
        )
        results = search_symbol(str(repo), "add")
        assert len(results) == 1
        assert results[0]["name"] == "add"
        assert results[0]["type"] == "function"


def test_syntax_check_ok_and_fail():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        good = repo / "good.py"
        good.write_text("x = 1 + 2\n")
        ok, msg = syntax_check(str(repo), "good.py")
        assert ok

        bad = repo / "bad.py"
        bad.write_text("def foo(\n")
        ok, msg = syntax_check(str(repo), "bad.py")
        assert not ok
        assert "SyntaxError" in msg


def test_write_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        success, _ = write_file(str(repo), "hello.py", "print('hi')\n")
        assert success
        content = read_file(str(repo), "hello.py")
        assert "print" in content
