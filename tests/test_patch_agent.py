import json
from pathlib import Path
from agent.patch_mode import PatchAgent,build_patch_messages,extract_patch

def test_patch_parser_and_ast_context():
 messages=build_patch_messages('add function is wrong',{'calc.py':'def add(a,b):\n return a-b\n','test_calc.py':'def test_x(): pass\n'})
 assert 'AST SYMBOL INDEX' in messages[1]['content']
 assert 'def add' in messages[1]['content']
 assert extract_patch('{"files":{"calc.py":"def add(a,b): return a+b"}}')['calc.py'].startswith('def add')

def test_patch_agent_applies_patch_and_requires_full_pytest():
 with __import__('tempfile').TemporaryDirectory() as td:
  root=Path(td);(root/'calc.py').write_text('def add(a,b):\n return a-b\n');(root/'test_calc.py').write_text('from calc import add\ndef test_add(): assert add(2,3)==5\n')
  class FakeModel:
   def __call__(self,messages):
    return json.dumps({'files':{'calc.py':'def add(a,b):\n return a+b\n'}})
  result=PatchAgent(root,'add must add',max_attempts=1).run(FakeModel())
  assert result['success'] is True
  assert result['events'][0]['tests']['passed'] is True
