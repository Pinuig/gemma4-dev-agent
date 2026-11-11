#!/usr/bin/env python3
"""Run a Gemma 4 base/SFT/GRPO patch model with AST context and test-driven retries."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.gemma4_backend import Gemma4PatchGenerator
from agent.patch_mode import PatchAgent

def main():
 p=argparse.ArgumentParser();p.add_argument('--model_path',required=True);p.add_argument('--adapter_path');p.add_argument('--repo_path',required=True);p.add_argument('--issue',required=True);p.add_argument('--max_attempts',type=int,default=3);p.add_argument('--time_budget_sec',type=int,default=600);p.add_argument('--max_new_tokens',type=int,default=1600);p.add_argument('--max_prompt_tokens',type=int,default=4096);p.add_argument('--trajectory_path');a=p.parse_args()
 model=Gemma4PatchGenerator(a.model_path,a.max_new_tokens,a.max_prompt_tokens,a.adapter_path)
 agent=PatchAgent(a.repo_path,a.issue,a.max_attempts,a.time_budget_sec,a.trajectory_path)
 result=agent.run(model);print(json.dumps(result,indent=2));raise SystemExit(0 if result['success'] else 1)
if __name__=='__main__':main()
