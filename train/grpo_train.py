"""Small, explicit group-relative policy-optimization loop for Gemma 4 text completions.

Dataset JSONL rows: {"prompt": str, "repo_files": {relative_path: full_text}}.
The model must return: {"files": {relative_path: full_replacement_text}}.
Each group samples G candidate patches for one issue; pytest execution supplies rewards.
This avoids relying on TRL model-class support for Gemma 4's multimodal wrapper.
"""
from __future__ import annotations
import argparse,json,math,os,re,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from agent.patch_mode import build_patch_messages

def completion_text(text):
 if isinstance(text,str): return text
 if isinstance(text,list): return ''.join(x.get('text','') if isinstance(x,dict) else str(x) for x in text)
 return str(text)
def extract_patch(text):
 m=re.search(r'\{.*\}',completion_text(text),re.S)
 if not m:return None
 try: obj=json.loads(m.group(0))
 except ValueError:return None
 files=obj.get('files') if isinstance(obj,dict) else None
 return files if isinstance(files,dict) and files else None

def execution_reward(repo_files,completion,timeout=45):
 patch=extract_patch(completion)
 if not isinstance(repo_files,dict) or not patch:return -0.5
 with tempfile.TemporaryDirectory(prefix='gemma4-grpo-') as td:
  root=Path(td).resolve()
  def put(rel,content):
   p=(root/rel).resolve()
   if root not in p.parents or not isinstance(content,str): raise ValueError('unsafe patch path/content')
   p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content,encoding='utf-8')
  try:
   for rel,content in repo_files.items():put(rel,content)
   for rel,content in patch.items():put(rel,content)
   # Reject syntax-invalid Python patches before paying for pytest.
   for rel in patch:
    if rel.endswith('.py'):
     p=(root/rel).resolve(); compile(p.read_text(encoding='utf-8'),str(p),'exec')
   env={**os.environ,'PYTHONPATH':str(root),'PYTHONDONTWRITEBYTECODE':'1'}
   result=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short'],cwd=root,capture_output=True,text=True,timeout=timeout,env=env)
   if result.returncode==0:return 1.0+min(0.1,0.02*len(patch))
   summary=result.stdout+'\n'+result.stderr
   passed=re.search(r'(\d+) passed',summary); failed=re.search(r'(\d+) failed',summary)
   n_pass=int(passed.group(1)) if passed else 0; n_fail=int(failed.group(1)) if failed else 0
   return max(-0.2,min(0.4,0.05+0.05*n_pass-0.05*n_fail))
  except SyntaxError:return -0.4
  except (ValueError,OSError):return -0.5
  except subprocess.TimeoutExpired:return -0.35

def render_prompt(processor,issue,repo_files):
 messages=build_patch_messages(issue,repo_files)
 return processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

def sequence_logprobs(model,input_ids,attention_mask,prompt_len,grad):
 ctx=torch.enable_grad() if grad else torch.no_grad()
 with ctx:
  logits=model(input_ids=input_ids,attention_mask=attention_mask).logits
  # logits[:, t-1] predicts input token at t
  logp=torch.log_softmax(logits[:,:-1,:].float(),dim=-1)
  targets=input_ids[:,1:]
  token_lp=logp.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
  start=max(0,prompt_len-1)
  return token_lp[:,start:]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--base_model',required=True); p.add_argument('--sft_adapter',required=True); p.add_argument('--data_path',required=True); p.add_argument('--output_dir',default='adapters/grpo'); p.add_argument('--steps',type=int,default=100); p.add_argument('--generations',type=int,default=4); p.add_argument('--lr',type=float,default=5e-6); p.add_argument('--max_prompt_tokens',type=int,default=4096); p.add_argument('--max_new_tokens',type=int,default=1200); p.add_argument('--temperature',type=float,default=0.8); p.add_argument('--clip_eps',type=float,default=0.2); p.add_argument('--ppo_epochs',type=int,default=2); p.add_argument('--load_in_4bit',action='store_true'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 try:
  from transformers import AutoProcessor,AutoModelForMultimodalLM,BitsAndBytesConfig
  from peft import PeftModel,prepare_model_for_kbit_training
 except ImportError as e: raise SystemExit('Install compatible torch, transformers>=5.10.1, accelerate, peft and bitsandbytes.') from e
 torch.manual_seed(a.seed)
 rows=[json.loads(x) for x in Path(a.data_path).read_text().splitlines() if x.strip()]
 if not rows:raise SystemExit('GRPO dataset is empty')
 for i,row in enumerate(rows):
  if not isinstance(row.get('prompt'),str) or not isinstance(row.get('repo_files'),dict):raise SystemExit(f'Row {i} must contain prompt:string and repo_files:object')
 proc=AutoProcessor.from_pretrained(a.base_model); tok=proc.tokenizer; tok.padding_side='left'
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16) if a.load_in_4bit else None
 model=AutoModelForMultimodalLM.from_pretrained(a.base_model,dtype='auto',device_map='auto',quantization_config=quant)
 if a.load_in_4bit: model=prepare_model_for_kbit_training(model)
 model=PeftModel.from_pretrained(model,a.sft_adapter,is_trainable=True); model.train()
 if hasattr(model,'gradient_checkpointing_enable'):model.gradient_checkpointing_enable()
 if hasattr(model,'config') and hasattr(model.config,'use_cache'):model.config.use_cache=False
 optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr)
 device=next(model.parameters()).device; history=[]
 for step in range(a.steps):
  row=rows[step%len(rows)]; prompt=render_prompt(proc,row.get('issue',row['prompt']),row['repo_files'])
  enc=tok(prompt,return_tensors='pt',truncation=True,max_length=a.max_prompt_tokens).to(device); prompt_ids=enc['input_ids']; prompt_len=prompt_ids.shape[1]
  candidates=[]; old_lps=[]; rewards=[]
  model.eval()
  for _ in range(a.generations):
   with torch.no_grad():
    seq=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=max(0.1,a.temperature),top_p=0.95,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
   full=seq[0]; completion_ids=full[prompt_len:]; text=tok.decode(completion_ids,skip_special_tokens=True); candidates.append((full.detach(),text))
   mask=torch.ones_like(full).unsqueeze(0)
   with torch.no_grad(): old=sequence_logprobs(model,full.unsqueeze(0),mask,prompt_len,False).squeeze(0).detach()
   old_lps.append(old); rewards.append(execution_reward(row['repo_files'],text))
  r=torch.tensor(rewards,dtype=torch.float32,device=device); std=r.std(unbiased=False)
  advantages=(r-r.mean())/(std+1e-4) if std.item()>1e-6 else torch.zeros_like(r)
  model.train()
  # Disable dropout during PPO ratio calculation so old/new policy log-probs align.
  for module in model.modules():
   if isinstance(module,torch.nn.Dropout): module.eval()
  losses=[]
  for _epoch in range(a.ppo_epochs):
   optimizer.zero_grad(set_to_none=True); epoch_losses=[]
   for j,(full,text) in enumerate(candidates):
    mask=torch.ones_like(full).unsqueeze(0); new=sequence_logprobs(model,full.unsqueeze(0),mask,prompt_len,True).squeeze(0)
    old=old_lps[j].to(new.device); n=min(new.numel(),old.numel()); new=new[:n]; old=old[:n]
    if n==0:continue
    ratio=torch.exp(torch.clamp(new-old,min=-10,max=10)); adv=advantages[j]
    unclipped=ratio*adv; clipped=torch.clamp(ratio,1-a.clip_eps,1+a.clip_eps)*adv
    loss=-torch.minimum(unclipped,clipped).mean()
    (loss/a.generations).backward(); epoch_losses.append(float(loss.detach().cpu()))
   torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.0); optimizer.step()
   losses.extend(epoch_losses)
  record={'step':step+1,'rewards':rewards,'mean_reward':float(r.mean().item()),'loss':sum(losses)/max(1,len(losses)),'ppo_epochs':a.ppo_epochs}; history.append(record)
  print(json.dumps(record),flush=True)
  if (step+1)%25==0:
   out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); proc.save_pretrained(out); (out/'grpo_metrics.json').write_text(json.dumps(history,indent=2))
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); proc.save_pretrained(out); (out/'grpo_metrics.json').write_text(json.dumps(history,indent=2)); print(f'GRPO adapter saved: {out}')
if __name__=='__main__':main()
