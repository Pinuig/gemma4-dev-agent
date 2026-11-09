import re
from agent.prompt import SYSTEM_PROMPT, SCRATCHPAD_TEMPLATE
from agent.tools import CodeTools

class DevAgentLoop:
    def __init__(self, repo_path: str, model_pipeline=None, max_steps: int = 5):
        self.repo_path = repo_path
        self.model = model_pipeline
        self.max_steps = max_steps
        self.history = ""

    def step(self, issue_description: str) -> bool:
        print(f"[*] Starting autonomous loop for: {issue_description}")
        observation = "Initial state. Ready for analysis."
        
        for iteration in range(1, self.max_steps + 1):
            prompt = SYSTEM_PROMPT + SCRATCHPAD_TEMPLATE.format(
                issue_description=issue_description, history=self.history, observation=observation
            )
            response = self.model(prompt, max_new_tokens=512) if self.model else "Thought: testing\nAction: run_tests()"
            self.history += f"\nIteration {iteration}:\n{response}"
            
            if "run_tests" in response:
                res = CodeTools.run_tests(self.repo_path)
                observation = f"Test Result: {'PASSED' if res['passed'] else 'FAILED'}\nLogs:\n{res['stdout'] or res['stderr']}"
                if res['passed']:
                    print("[✓] Bug successfully resolved!")
                    return True
            elif "read_file" in response:
                match = re.search(r"read_file\(['\"](.*?)['\"]\)", response)
                observation = CodeTools.read_file(self.repo_path, match.group(1) if match else "test_sample.py")
            elif "search_symbol" in response:
                match = re.search(r"search_symbol\(['\"](.*?)['\"]\)", response)
                observation = CodeTools.search_symbol(self.repo_path, match.group(1) if match else "add_numbers")
            else:
                observation = "Action executed."
        return False
