"""Supervised fine-tuning for Gemma 4 using chat trajectory JSONL and LoRA.
Each JSONL row: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]}.
"""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model_name',required=True); p.add_argument('--data_path',required=True); p.add_argument('--output_dir',default='adapters/sft'); p.add_argument('--epochs',type=float,default=2); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--max_length',type=int,default=4096); p.add_argument('--batch_size',type=int,default=1); p.add_argument('--grad_accum',type=int,default=8); p.add_argument('--lora_r',type=int,default=16); p.add_argument('--load_in_4bit',action='store_true'); args=p.parse_args()
 try:
  import torch
  from datasets import Dataset
  from transformers import AutoProcessor,AutoModelForMultimodalLM,Trainer,TrainingArguments
  from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
 except ImportError as e: raise SystemExit('Install compatible torch, transformers>=5.10.1, datasets, peft, accelerate.') from e
 rows=[json.loads(x) for x in Path(args.data_path).read_text().splitlines() if x.strip()]
 if not rows: raise SystemExit('No training examples found.')
 processor=AutoProcessor.from_pretrained(args.model_name); tokenizer=processor.tokenizer
 if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
 model=AutoModelForMultimodalLM.from_pretrained(args.model_name,dtype='auto',device_map='auto',load_in_4bit=args.load_in_4bit)
 if args.load_in_4bit: model=prepare_model_for_kbit_training(model)
 lora=LoraConfig(r=args.lora_r,lora_alpha=args.lora_r*2,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules='all-linear')
 model=get_peft_model(model,lora); model.print_trainable_parameters()
 encoded_rows=[]
 for i,row in enumerate(rows):
  if 'messages' not in row: raise ValueError(f'Row {i} missing messages')
  messages=row['messages']
  full_ids=tokenizer(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False),truncation=True,max_length=args.max_length)['input_ids']
  labels=[-100]*len(full_ids)
  # Supervise assistant turns only; system/user/tool observations are context.
  for j,msg in enumerate(messages):
   if msg.get('role')!='assistant': continue
   prefix=messages[:j]
   prefix_text=processor.apply_chat_template(prefix,tokenize=False,add_generation_prompt=True)
   end_text=processor.apply_chat_template(messages[:j+1],tokenize=False,add_generation_prompt=False)
   begin=len(tokenizer(prefix_text,truncation=True,max_length=args.max_length)['input_ids'])
   end=min(len(full_ids),len(tokenizer(end_text,truncation=True,max_length=args.max_length)['input_ids']))
   for k in range(min(begin,len(labels)),end): labels[k]=full_ids[k]
  if not any(x!=-100 for x in labels): raise ValueError(f'Row {i} contains no assistant tokens after truncation')
  encoded_rows.append((full_ids,labels))
 class ChatDataset(torch.utils.data.Dataset):
  def __len__(self): return len(encoded_rows)
  def __getitem__(self,i):
   ids,labels=encoded_rows[i]; pad=args.max_length-len(ids)
   return {'input_ids':torch.tensor(ids+[tokenizer.pad_token_id or tokenizer.eos_token_id]*pad,dtype=torch.long),'attention_mask':torch.tensor([1]*len(ids)+[0]*pad,dtype=torch.long),'labels':torch.tensor(labels+[-100]*pad,dtype=torch.long)}
 out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 train_args=TrainingArguments(output_dir=str(out),num_train_epochs=args.epochs,per_device_train_batch_size=args.batch_size,gradient_accumulation_steps=args.grad_accum,learning_rate=args.lr,logging_steps=5,save_strategy='epoch',gradient_checkpointing=True,fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),report_to='none',remove_unused_columns=False)
 trainer=Trainer(model=model,args=train_args,train_dataset=ChatDataset()); trainer.train(); model.save_pretrained(out); processor.save_pretrained(out)
 print(f'SFT adapter saved to {out}')
if __name__=='__main__': main()
