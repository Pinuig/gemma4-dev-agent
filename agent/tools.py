import ast
import os
import subprocess

class CodeTools:
    @staticmethod
    def search_symbol(repo_path: str, symbol_name: str) -> str:
        matches = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            tree = ast.parse(f.read(), filename=full_path)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                if node.name == symbol_name:
                                    rel_path = os.path.relpath(full_path, repo_path)
                                    matches.append(f"File: {rel_path} | Line: {node.lineno} | Type: {type(node).__name__}")
                    except Exception:
                        continue
        return "\n".join(matches) if matches else f"Symbol '{symbol_name}' not found."

    @staticmethod
    def read_file(repo_path: str, relative_path: str) -> str:
        full_path = os.path.join(repo_path, relative_path)
        if not os.path.exists(full_path):
            return f"Error: File '{relative_path}' does not exist."
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def apply_patch(repo_path: str, target_file: str, patch_content: str) -> str:
        full_path = os.path.join(repo_path, target_file)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(patch_content)
            return f"Successfully applied changes to '{target_file}'."
        except Exception as e:
            return f"Failed to apply patch: {str(e)}"

    @staticmethod
    def run_tests(repo_path: str) -> dict:
        try:
            result = subprocess.run(
                ["pytest", "--tb=short"], cwd=repo_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
            )
            return {"passed": result.returncode == 0, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"passed": False, "returncode": -1, "stdout": "", "stderr": str(e)}
