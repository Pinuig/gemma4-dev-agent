"""Shared AST-context prompt, patch parser, and verified direct-patch controller."""
from __future__ import annotations
import ast,json,re
from pathlib import Path
from .tools import _root,write_file,syntax_check,run_pytest

PATCH_SYSTEM='''You are Gemma 4, an expert Python repository repair model. You will receive a bug report, an AST summary, and selected repository files. Return ONLY valid JSON with this exact shape: {"files":{"relative/path.py":"complete replacement file contents"}}. Include only files that must change. Do not use markdown fences or explanatory text. Do not change tests to hide defects. Preserve unrelated behavior. Your patch will be applied to a clean working copy and checked with Python syntax validation and the full pytest suite.'''

def _rank_file(issue,rel,content):
 words={x.lower() for x in re.findall(r'[A-Za-z_][A-Za-z_0-9]{2,}',issue)}
 text=(rel+' '+content[:12000]).lower()
 score=sum(1 for w in words if w in text)
 if 'test_' in rel or '/tests/' in '/'+rel:score+=4
 if rel.endswith('.py'):score+=1
 return score

def build_patch_messages(issue,repo_files,max_context_chars=26000):
 defs=[]
 for rel,src in repo_files.items():
  if not rel.endswith('.py'):continue
  try:
   tree=ast.parse(src)
   for node in ast.walk(tree):
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
     defs.append(f"{rel}:{node.lineno} {type(node).__name__} {node.name}")
  except SyntaxError:defs.append(f'{rel}: [SYNTAX ERROR]')
 ranked=sorted(repo_files.items(),key=lambda x:(-_rank_file(issue,x[0],x[1]),x[0]))
 selected=[]; used=0
 for rel,content in ranked:
  block=f'\n### FILE: {rel}\n{content}\n'
  if used+len(block)>max_context_chars:
   if not selected and len(block)>max_context_chars:block=block[:max_context_chars]
   else:continue
  selected.append(block);used+=len(block)
 ast_text='\n'.join(defs[:500]) or '(No Python AST definitions found.)'
 user=f'ISSUE:\n{issue}\n\nAST SYMBOL INDEX:\n{ast_text}\n\nREPOSITORY FILES:\n'+''.join(selected)
 return [{'role':'system','content':PATCH_SYSTEM},{'role':'user','content':user}]

def extract_patch(text):
 text=str(text); match=re.search(r'\{.*\}',text,re.S)
 if not match:return None
 try:obj=json.loads(match.group(0))
 except ValueError:return None
 files=obj.get('files') if isinstance(obj,dict) else None
 if not isinstance(files,dict) or not files:return None
 if any(not isinstance(k,str) or not isinstance(v,str) for k,v in files.items()):return None
 return files

def snapshot_repo(repo_path,max_files=250,max_total_chars=400000):
 root=_root(repo_path); files={}; total=0
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if any(part.startswith('.') or part in {'venv','.venv','__pycache__','.pytest_cache','node_modules'} for part in rel.parts):continue
  if p.suffix.lower() not in {'.py','.toml','.ini','.cfg','.txt','.yaml','.yml','.json'}:continue
  try:content=p.read_text(encoding='utf-8')
  except (UnicodeDecodeError,OSError):continue
  if len(files)>=max_files or total+len(content)>max_total_chars:break
  files[str(rel)]=content;total+=len(content)
 return files

class PatchAgent:
 def __init__(self,repo_path,issue,max_attempts=3,time_budget_sec=600,trajectory_path=None):
  self.repo_path=str(Path(repo_path).resolve());self.issue=issue;self.max_attempts=max_attempts;self.time_budget_sec=time_budget_sec;self.trajectory_path=trajectory_path
 def run(self,generate_fn):
  import time
  start=time.monotonic();initial=snapshot_repo(self.repo_path);events=[];last_error=''
  for attempt in range(self.max_attempts):
   if time.monotonic()-start>self.time_budget_sec:break
   messages=build_patch_messages(self.issue+('\n\nPREVIOUS ATTEMPT RESULT:\n'+last_error if last_error else ''),snapshot_repo(self.repo_path))
   response=generate_fn(messages);patch=extract_patch(response)
   if not patch:last_error='Model did not return valid JSON with a non-empty files object.';events.append({'attempt':attempt+1,'valid_patch':False,'error':last_error});continue
   applied=[];error=None
   for rel,content in patch.items():
    try: result=write_file(self.repo_path,rel,content)
    except Exception as exc: result={'ok':False,'error':f'{type(exc).__name__}: {exc}'}
    if not result.get('ok'):error=result;break
    applied.append(rel)
   syntax=[]
   if not error:
    for rel in applied:
     if rel.endswith('.py'):
      result=syntax_check(self.repo_path,rel);syntax.append({'file':rel,**result})
      if not result.get('ok'):error=result;break
   tests=run_pytest(self.repo_path,timeout=90) if not error else {'passed':False,'summary':'SYNTAX_OR_WRITE_FAILURE','output':str(error)}
   event={'attempt':attempt+1,'patch':patch,'applied':applied,'syntax':syntax,'tests':tests};events.append(event)
   if tests.get('passed'):
    result={'success':True,'message':'All pytest tests passed after the latest patch.','attempts':attempt+1,'elapsed_sec':round(time.monotonic()-start,2),'events':events}
    self._save(initial,patch,result);return result
   last_error=json.dumps({'syntax':syntax,'tests':tests},ensure_ascii=False)[:6000]
  result={'success':False,'message':'No verified passing patch within the attempt/time budget.','attempts':len(events),'elapsed_sec':round(time.monotonic()-start,2),'events':events}
  self._save(initial,{},result);return result
 def _save(self,initial,patch,result):
  if not self.trajectory_path:return
  p=Path(self.trajectory_path);p.parent.mkdir(parents=True,exist_ok=True)
  p.write_text(json.dumps({'issue':self.issue,'initial_files':initial,'final_patch':patch,'result':result},ensure_ascii=False,indent=2),encoding='utf-8')
