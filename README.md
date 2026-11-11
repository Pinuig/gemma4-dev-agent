# AST-Guided Self-Correcting Coding Agent on Gemma 4

Research implementation with two compatible execution paths: a base Gemma 4 tool-calling agent for trajectory collection, and a direct JSON-patch agent used consistently for SFT, execution-reward GRPO, and adapter evaluation. Both use deterministic AST context and verified pytest feedback.

> **Research honesty:** this repository is runnable code, not evidence that training has already occurred or that the agent fixes every issue. You must run it with a compatible Gemma 4 checkpoint, collect real trajectories, train adapters, and report measured results. The included tiny examples are smoke tests only.

## 1. Environment

Use the Kaggle competition notebook GPU or a compatible CUDA machine. Gemma 4 multimodal checkpoints require a compatible recent Transformers build (>=5.10.1 per Gemma 4 docs). Python 3.11/3.12 is recommended for ML dependency compatibility; the user's local Python 3.14 environment may not have wheels for every dependency.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch matching the CUDA runtime first, then:
pip install -r requirements.txt
```

Confirm GPU and library versions:
```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

## 2. Run tests and the actual Gemma agent

```bash
python -m pytest -v
cp -a examples/demo_repo /tmp/gemma4-demo-repo
```

Run with a local checkpoint directory (for Kaggle, use the mounted `/kaggle/input/...` path):
```bash
python scripts/run_gemma4.py \
  --model_path /path/to/gemma-4-checkpoint \
  --repo_path /tmp/gemma4-demo-repo \
  --issue "calculator.add subtracts instead of adding. Fix it and run all pytest tests." \
  --max_turns 24 --max_tool_calls 32 --time_budget_sec 600 \
  --trajectory_path runs/trajectories/demo.json
```

The tool-calling path above is used to collect grounded trajectories. The SFT/GRPO adapters are trained for direct JSON-patch generation, so run those adapters with the matching patch-mode controller:
```bash
python scripts/run_gemma4_patch.py --model_path /path/to/base-gemma4 \
  --adapter_path adapters/grpo --repo_path /tmp/gemma4-demo-repo \
  --issue "..." --trajectory_path runs/trajectories/patch-task.json
```

**Important:** run only trusted repositories. `run_pytest` executes project code in a subprocess with a timeout; it is not a hardened sandbox. For untrusted code, use a disposable isolated container/VM with no secrets, no network, and strict resource limits.

## 3. Collect successful trajectories

Run the agent on multiple real issues and save each run with `--trajectory_path`. Keep only successful, verified trajectories for SFT:
```bash
python scripts/build_sft_dataset.py --input_dir runs/trajectories --output data/sft.jsonl
```
The included `data/example_sft.jsonl` only verifies the data format. It is not enough to train a useful model.

## 4. SFT

Use a base Gemma 4 checkpoint and a diverse JSONL of successful conversations:
```bash
python train/sft_train.py \
  --model_name /path/to/gemma-4-checkpoint \
  --data_path data/sft.jsonl \
  --output_dir adapters/sft \
  --epochs 2 --batch_size 1 --grad_accum 8
```
This saves a PEFT LoRA adapter. Competition QAT-W4A16 checkpoints are already quantized; do not add `--load_in_4bit` unless you have confirmed that the checkpoint expects BitsAndBytes quantization. For a non-quantized base checkpoint, `--load_in_4bit` is an optional memory-saving mode if supported by the architecture.

## 5. GRPO with execution-based reward

Create a task manifest with one JSONL row per trusted issue, containing `issue` and `repo_path`. The builder snapshots source and test files; the GRPO prompt then includes the AST symbol index and selected repository file contents, matching the patch-mode inference prompt.

```bash
python scripts/build_grpo_dataset.py --tasks data/tasks.jsonl --output data/grpo_tasks.jsonl
```
Example smoke-test dataset: `data/example_grpo.jsonl`. It is intentionally tiny and only a format example.

```bash
python train/grpo_train.py \
  --base_model /path/to/gemma-4-checkpoint \
  --sft_adapter adapters/sft \
  --data_path data/grpo_tasks.jsonl \
  --output_dir adapters/grpo \
  --steps 100 --generations 4
```
The custom GRPO loop samples multiple patch completions per issue, runs each patch against a clean repository snapshot, normalizes rewards within the group, and applies a clipped sequence-level policy objective. Positive reward requires actual test success; malformed patches, test failures, syntax errors and timeouts are penalized. This is expensive: each candidate launches pytest. Use trusted, deterministic tests only. The optional `--load_in_4bit` flag is for compatible non-quantized base checkpoints; omit it for pre-quantized QAT checkpoints unless you have verified the loader behavior. This is a compact research implementation of group-relative policy optimization, not a drop-in replacement for every feature of a full PPO/GRPO framework; validate loss behavior and training stability before drawing research conclusions.

## 6. Evaluation

Prepare benchmark JSONL rows with `issue` and `repo_path` (each repository contains its tests). The evaluator copies each repo to a temporary directory before allowing edits:
```bash
python scripts/evaluate_agent.py \
  --model_path /path/to/gemma-4-checkpoint \
  --benchmark data/benchmark.jsonl --output results/base.json

python scripts/evaluate_agent.py \
  --model_path /path/to/gemma-4-checkpoint --adapter_path adapters/sft \
  --benchmark data/benchmark.jsonl --output results/sft.json

python scripts/evaluate_agent.py \
  --model_path /path/to/gemma-4-checkpoint --adapter_path adapters/grpo \
  --benchmark data/benchmark.jsonl --output results/sft_grpo.json
```

Report at least: resolved tasks / total, resolution rate, mean patch attempts, mean wall-clock time, and per-task outcomes. Use the same held-out tasks, attempt/time budgets, and hardware for all comparisons. Do not train on evaluation tasks.

## Architecture

`issue -> repository_map(AST) -> symbol/file inspection -> patch -> syntax check -> pytest -> failure evidence -> repair -> verified success`

Training stages:
`successful tool trajectories -> direct-patch SFT LoRA -> AST-context patch GRPO -> held-out patch-agent evaluation`

## Limitations to disclose in a paper

- AST indexing currently targets Python source; non-Python languages need language-specific parsers.
- The agent's file writer performs complete-file replacement, not general diff application.
- Test execution is not a security sandbox.
- GRPO reward quality depends on deterministic, meaningful tests; passing visible tests does not prove correctness or prevent overfitting.
- The included sample datasets are demonstrations, not a statistically meaningful benchmark.
- The Gemma 4/PEFT multimodal training integration must be smoke-tested against the exact checkpoint and package versions used in the experiment.
