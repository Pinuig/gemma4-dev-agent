"""Gemma 4 Transformers backend. Model generates tool calls; controller executes them."""
from __future__ import annotations
import json
TOOL_SCHEMAS=[
 {'type':'function','function':{'name':'repository_map','description':'Map Python files, AST definitions, imports and references.','parameters':{'type':'object','properties':{'max_files':{'type':'integer'}},'required':[]}}},
 {'type':'function','function':{'name':'search_symbol','description':'Find Python definitions by partial name using AST.','parameters':{'type':'object','properties':{'symbol_name':{'type':'string'},'file_glob':{'type':'string'}},'required':['symbol_name']}}},
 {'type':'function','function':{'name':'read_file','description':'Read a repository-relative file.','parameters':{'type':'object','properties':{'rel_path':{'type':'string'},'max_lines':{'type':'integer'}},'required':['rel_path']}}},
 {'type':'function','function':{'name':'write_file','description':'Replace complete contents of an inspected repository-relative file.','parameters':{'type':'object','properties':{'rel_path':{'type':'string'},'content':{'type':'string'}},'required':['rel_path','content']}}},
 {'type':'function','function':{'name':'syntax_check','description':'Compile-check a Python file.','parameters':{'type':'object','properties':{'rel_path':{'type':'string'}},'required':['rel_path']}}},
 {'type':'function','function':{'name':'run_pytest','description':'Run pytest in the repository with a timeout.','parameters':{'type':'object','properties':{'test_path':{'type':'string'},'keyword':{'type':'string'},'timeout':{'type':'integer'}},'required':[]}}},
]
class Gemma4Generator:
 def __init__(self,model_path,max_new_tokens=768,enable_thinking=False,trust_remote_code=False,adapter_path=None):
  try:
   import torch
   from transformers import AutoModelForMultimodalLM,AutoProcessor
  except ImportError as e: raise RuntimeError('Install a Gemma-4-compatible Transformers build (>=5.10.1), PyTorch and Accelerate.') from e
  self.torch=torch; self.max_new_tokens=max_new_tokens; self.enable_thinking=enable_thinking
  self.processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=trust_remote_code)
  self.model=AutoModelForMultimodalLM.from_pretrained(model_path,dtype='auto',device_map='auto',trust_remote_code=trust_remote_code)
  if adapter_path:
   try:
    from peft import PeftModel
    self.model=PeftModel.from_pretrained(self.model,adapter_path)
   except ImportError as e: raise RuntimeError('Install peft to load a LoRA adapter.') from e
  self.model.eval()
 def __call__(self,messages):
  prompt=self.processor.apply_chat_template(messages,tools=TOOL_SCHEMAS,tokenize=False,add_generation_prompt=True,enable_thinking=self.enable_thinking)
  inputs=self.processor(text=prompt,return_tensors='pt').to(self.model.device)
  with self.torch.inference_mode():
   ids=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
  raw=self.processor.decode(ids[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=False)
  return self._normalise(raw)
 def _normalise(self,raw):
  parser=getattr(self.processor,'parse_response',None); parsed={}
  if callable(parser):
   try: parsed=parser(raw,prefix='') or {}
   except Exception: pass
  calls=parsed.get('tool_calls') or []
  if calls:
   c=calls[0]; f=c.get('function',c); name=f.get('name'); args=f.get('arguments',{})
   if isinstance(args,str):
    try: args=json.loads(args)
    except ValueError: return 'Tool call arguments were invalid JSON. Retry.'
   if isinstance(name,str) and isinstance(args,dict): return '<tool_call>'+json.dumps({'name':name,'arguments':args},ensure_ascii=False)+'</tool_call>'
  content=parsed.get('content')
  if isinstance(content,str) and content.strip(): return content.strip()
  cleaned=raw.replace('<|im_end|>','').strip()
  return cleaned or 'Please call an available tool or provide a verified final result.'

class Gemma4PatchGenerator:
 """Text-only patch completion backend for SFT/GRPO adapters trained on JSON patches."""
 def __init__(self,model_path,max_new_tokens=1600,max_prompt_tokens=4096,adapter_path=None,trust_remote_code=False):
  try:
   import torch
   from transformers import AutoModelForMultimodalLM,AutoProcessor
  except ImportError as exc: raise RuntimeError('Install Gemma-4-compatible Transformers >=5.10.1, PyTorch and Accelerate.') from exc
  self.torch=torch;self.max_new_tokens=max_new_tokens;self.max_prompt_tokens=max_prompt_tokens
  self.processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=trust_remote_code)
  self.model=AutoModelForMultimodalLM.from_pretrained(model_path,dtype='auto',device_map='auto',trust_remote_code=trust_remote_code)
  if adapter_path:
   from peft import PeftModel
   self.model=PeftModel.from_pretrained(self.model,adapter_path)
  self.model.eval()
 def __call__(self,messages):
  prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=self.processor(text=prompt,return_tensors='pt',truncation=True,max_length=self.max_prompt_tokens).to(self.model.device)
  with self.torch.inference_mode():
   ids=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
  return self.processor.decode(ids[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
