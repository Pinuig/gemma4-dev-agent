import tempfile
from pathlib import Path
import pytest
from agent.tools import search_symbol,syntax_check,write_file,read_file,repository_map

def test_search_symbol_finds_function():
 with tempfile.TemporaryDirectory() as tmp:
  repo=Path(tmp); (repo/'math_utils.py').write_text('def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n')
  results=search_symbol(str(repo),'add'); assert len(results)==1; assert results[0]['name']=='add'; assert results[0]['kind']=='function'
def test_syntax_check_ok_and_fail():
 with tempfile.TemporaryDirectory() as tmp:
  repo=Path(tmp); (repo/'good.py').write_text('x = 1 + 2\n'); assert syntax_check(str(repo),'good.py')['ok']
  (repo/'bad.py').write_text('def foo(\n'); result=syntax_check(str(repo),'bad.py'); assert not result['ok']; assert 'SyntaxError' not in result['message']
def test_write_and_read():
 with tempfile.TemporaryDirectory() as tmp:
  assert write_file(tmp,'hello.py',"print('hi')\n")['ok']; assert 'print' in read_file(tmp,'hello.py')
def test_reject_path_traversal():
 with tempfile.TemporaryDirectory() as tmp:
  with pytest.raises(ValueError): write_file(tmp,'../escape.py','bad')
def test_repository_map_extracts_class_and_imports():
 with tempfile.TemporaryDirectory() as tmp:
  repo=Path(tmp); (repo/'a.py').write_text('import os\nclass Box:\n    def open(self):\n        return os.getcwd()\n')
  result=repository_map(tmp); assert 'a.py' in result['python_files']; assert any(x['name']=='Box' for x in result['symbols']); assert result['imports']['a.py']==['os']
