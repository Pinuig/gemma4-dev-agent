"""Convert successful tool-agent trajectories into direct patch-mode SFT examples."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.patch_mode import build_patch_messages

def main():
 p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output',default='data/sft_trajectories.jsonl');a=p.parse_args();out=[]
 for path in Path(a.input_dir).rglob('*.json'):
  try:row=json.loads(path.read_text())
  except Exception:continue
  if not row.get('result',{}).get('success'):continue
  initial=row.get('initial_files')
  patch=row.get('final_patch') or {}
  for event in row.get('events',[]):
   if event.get('tool')=='write_file':
    args=event.get('arguments',{})
    if isinstance(args.get('rel_path'),str) and isinstance(args.get('content'),str):patch[args['rel_path']]=args['content']
  if not isinstance(initial,dict) or not initial or not patch:continue
  messages=build_patch_messages(row.get('issue',''),initial)
  messages.append({'role':'assistant','content':json.dumps({'files':patch},ensure_ascii=False)})
  out.append({'messages':messages})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',encoding='utf-8') as f:
  for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(f'Wrote {len(out)} successful patch trajectories to {a.output}')
if __name__=='__main__':main()
