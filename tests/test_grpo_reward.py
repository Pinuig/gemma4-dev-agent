import json
from pathlib import Path
from train.grpo_train import execution_reward,extract_patch

def test_extract_patch():
 assert extract_patch('{"files":{"a.py":"x=1"}}') == {'a.py':'x=1'}
 assert extract_patch('not json') is None

def test_execution_reward_requires_tests_to_pass():
 repo=Path(__file__).resolve().parents[1]/'examples'/'demo_repo'
 files={p.name:p.read_text() for p in repo.glob('*.py')}
 good='{"files":{"calculator.py":"def add(a, b):\\n    return a + b\\n"}}'
 bad='{"files":{"calculator.py":"def add(a, b):\\n    return a - b\\n"}}'
 assert execution_reward(files,good)>0
 assert execution_reward(files,bad)<0
