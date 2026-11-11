"""Build GRPO task JSONL snapshots from a task list of {issue, repo_path}."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--tasks',required=True,help='Input JSONL with issue and repo_path');p.add_argument('--output',default='data/grpo_tasks.jsonl');a=p.parse_args();out=[]
 for line_no,line in enumerate(Path(a.tasks).read_text().splitlines(),1):
  if not line.strip():continue
  row=json.loads(line); repo=Path(row['repo_path']).resolve()
  if not repo.is_dir():raise SystemExit(f'Line {line_no}: repo_path is not a directory: {repo}')
  files={}
  for f in repo.rglob('*'):
   if not f.is_file() or any(x in {'.git','.venv','venv','__pycache__','.pytest_cache','node_modules'} or x.startswith('.') for x in f.relative_to(repo).parts):continue
   if f.name.lower() in {'id_rsa','id_ed25519','credentials.json','service-account.json'} or f.suffix.lower() in {'.pem','.key','.p12','.pfx'}:continue
   if f.stat().st_size>300_000:continue
   try: files[str(f.relative_to(repo))]=f.read_text(encoding='utf-8')
   except (UnicodeDecodeError,OSError):continue
  if not any(x.startswith('test_') or '/test_' in x for x in files):raise SystemExit(f'Line {line_no}: repository snapshot contains no test_*.py tests')
  issue=row['issue'];prompt='Fix this issue. Output ONLY JSON: {"files":{"relative/path.py":"complete replacement file content"}}. Do not use markdown.\nISSUE: '+issue
  out.append({'prompt':prompt,'issue':issue,'repo_files':files})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',encoding='utf-8') as f:
  for row in out:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(f'Wrote {len(out)} GRPO task snapshots to {a.output}')
if __name__=='__main__':main()
