"""Budgeted, trajectory-recording Gemma coding-agent controller."""
from __future__ import annotations
import json,time
from pathlib import Path
from .tools import call_tool
from .prompt import build_system_prompt,parse_tool_call
from .patch_mode import snapshot_repo

class DevAgent:
 def __init__(self,repo_path,issue,max_turns=24,max_tool_calls=32,time_budget_sec=600,trajectory_path=None):
  self.repo_path=str(Path(repo_path).resolve()); self.issue=issue; self.max_turns=max_turns; self.max_tool_calls=max_tool_calls; self.time_budget_sec=time_budget_sec; self.trajectory_path=trajectory_path; self.events=[]
 def run(self,model_generate_fn):
  start=time.monotonic(); self.initial_files=snapshot_repo(self.repo_path); messages=[{'role':'system','content':build_system_prompt(self.issue,self.repo_path)}]; calls=0; last_full_test=False
  for turn in range(self.max_turns):
   if calls>=self.max_tool_calls or time.monotonic()-start>=self.time_budget_sec: break
   response=model_generate_fn(messages); messages.append({'role':'assistant','content':response})
   if response.strip().startswith('SUCCESS:'):
    if last_full_test:
     return self._finish(True,response,calls,start,messages)
    messages.append({'role':'user','content':'You cannot claim success yet. Run pytest after the latest edits and verify it passes.'}); continue
   name,args=parse_tool_call(response)
   if not name:
    messages.append({'role':'user','content':'Invalid or missing tool call. Use the exact JSON <tool_call>{"name":"...","arguments":{...}}</tool_call> format, or explain a blocker. Do not claim success without passing tests.'}); continue
   calls+=1
   if name=='write_file': last_full_test=False
   result=call_tool(name,self.repo_path,**args)
   if name=='run_pytest' and not args.get('test_path') and not args.get('keyword') and isinstance(result,dict): last_full_test=bool(result.get('passed'))
   event={'turn':turn,'tool':name,'arguments':args,'result':result}; self.events.append(event)
   messages.append({'role':'user','content':'TOOL_RESULT '+json.dumps(event,ensure_ascii=False,default=str)[:7000]})
  return self._finish(False,'Budget exhausted or no verified success',calls,start,messages)
 def _finish(self,success,message,calls,start,messages):
  result={'success':success,'message':message,'tool_calls':calls,'elapsed_sec':round(time.monotonic()-start,2),'events':self.events,'messages':messages}
  if self.trajectory_path:
   p=Path(self.trajectory_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'issue':self.issue,'initial_files':self.initial_files,'events':self.events,'messages':messages,'result':{k:v for k,v in result.items() if k!='messages'}},ensure_ascii=False,indent=2),encoding='utf-8')
  return result
