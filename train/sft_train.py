import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

def train_sft():
    model, tokenizer = FastLanguageModel.from_pretrained(model_name="google/gemma-4-7b", max_seq_length=2048, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_alpha=16)
    print("[✓] SFT training script initialized.")

if __name__ == "__main__":
    train_sft()
