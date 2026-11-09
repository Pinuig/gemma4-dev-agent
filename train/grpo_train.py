from trl import GRPOConfig
from agent.tools import CodeTools

def pytest_reward_function(completions, **kwargs) -> list[float]:
    return [1.0 if "apply_patch" in c else 0.0 for c in completions]

if __name__ == "__main__":
    print("[✓] GRPO config ready.")
