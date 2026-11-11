"""Run a Gemma 4 checkpoint/adapter against a JSONL issue benchmark and report metrics."""
import argparse,json,shutil,tempfile,sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agent.patch_mode import PatchAgent
from agent.gemma4_backend import Gemma4PatchGenerator

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model_path',required=True); p.add_argument('--adapter_path'); p.add_argument('--benchmark',required=True,help='JSONL rows: issue, repo_path'); p.add_argument('--output',default='results/evaluation.json'); p.add_argument('--max_attempts',type=int,default=3); p.add_argument('--time_budget_sec',type=int,default=600); p.add_argument('--max_new_tokens',type=int,default=1600); p.add_argument('--max_prompt_tokens',type=int,default=4096); a=p.parse_args()
 model=Gemma4PatchGenerator(a.model_path,max_new_tokens=a.max_new_tokens,max_prompt_tokens=a.max_prompt_tokens,adapter_path=a.adapter_path); rows=[]
 for line in Path(a.benchmark).read_text().splitlines():
  if not line.strip():continue
  task=json.loads(line); source=Path(task['repo_path']).resolve()
  with tempfile.TemporaryDirectory(prefix='gemma4-eval-') as td:
   repo=Path(td)/'repo'; shutil.copytree(source,repo,ignore=shutil.ignore_patterns('.git','.venv','venv','__pycache__','.pytest_cache'))
   agent=PatchAgent(repo,task['issue'],a.max_attempts,a.time_budget_sec)
   result=agent.run(model); rows.append({'issue':task['issue'],'success':result['success'],'attempts':result['attempts'],'elapsed_sec':result['elapsed_sec'],'message':result['message']})
 metrics={'tasks':len(rows),'resolved':sum(x['success'] for x in rows),'resolve_rate':sum(x['success'] for x in rows)/len(rows) if rows else 0,'mean_attempts':sum(x['attempts'] for x in rows)/len(rows) if rows else 0,'mean_elapsed_sec':sum(x['elapsed_sec'] for x in rows)/len(rows) if rows else 0,'results':rows}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
