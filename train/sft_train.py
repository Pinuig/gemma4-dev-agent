"""
Supervised Fine-Tuning of Gemma 4 with Unsloth + QLoRA on code-editing trajectories.

Usage (Kaggle L4x4 or local with enough VRAM): python train/sft_train.py --data_path data/trajectories.jsonl --output_dir adapters/sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--data_path", required=True, help="JSONL with {messages: [...]} or {prompt, completion}")
    parser.add_argument("--output_dir", default="adapters/sft_lora")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError as e:
        print("Missing dependency. Install with:")
        print("  pip install unsloth transformers trl datasets peft accelerate")
        raise SystemExit(1) from e

    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_dataset("json", data_files=args.data_path, split="train")

    def formatting_func(examples):
        if "text" in examples:
            return examples["text"]
        if "messages" in examples:
            return [tokenizer.apply_chat_template(m, tokenize=False) for m in examples["messages"]]
        if "prompt" in examples and "completion" in examples:
            return [p + c for p, c in zip(examples["prompt"], examples["completion"])]
        raise ValueError("Dataset must contain 'text', 'messages' or 'prompt'+'completion'")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",          
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_func if "messages" in ds.column_names or "prompt" in ds.column_names else None,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
    )

    print("Starting SFT...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
