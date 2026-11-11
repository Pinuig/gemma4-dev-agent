"""
Group Relative Policy Optimization (GRPO) with execution reward.
Uses TRL's GRPOTrainer. Reward is derived from local pytest outcomes.

This is a research reference implementation. In practice you will need:
- A set of training tasks (issue + repo snapshot)
- A way to run the agent loop and obtain the final pytest result
- Careful batching because each reward evaluation is expensive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def pytest_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Placeholder reward function.
    In a real run you would:
    1. Parse the completion for the final patch / write_file calls
    2. Apply them to a clean copy of the repo
    3. Run the relevant pytest
    4. Return 1.0 if passed, 0.0 otherwise (or shaped reward)
    """
    rewards = []
    for comp in completions:
        if "SUCCESS:" in comp or "PASSED" in comp:
            rewards.append(1.0)
        elif "SyntaxError" in comp or "ERROR" in comp:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="adapters/sft_lora")
    parser.add_argument("--output_dir", default="adapters/grpo_lora")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=4)  # group size
    args = parser.parse_args()

    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
    except ImportError as e:
        print("Install TRL >= 0.12 and transformers.")
        raise SystemExit(1) from e

    print("Loading SFT model for GRPO...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dummy dataset of prompts (replace with real issue descriptions)
    prompts = [
        "Fix the off-by-one error in the binary search function so that test_binary_search passes.",
        "The add function returns the wrong sum when both arguments are negative. Make the tests green.",
        "Handle the empty-list edge case in average() without raising ZeroDivisionError.",
    ]
    dataset = Dataset.from_dict({"prompt": prompts})

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=pytest_reward,
        processing_class=tokenizer,
    )

    print("Starting GRPO (this will be slow because reward evaluation runs code)...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"GRPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
