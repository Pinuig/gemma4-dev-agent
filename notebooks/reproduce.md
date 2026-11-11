# Gemma 4 DevAgent reproducibility notes

This project uses Hugging Face Transformers directly. It does not require Unsloth or TRL for the supplied SFT/GRPO scripts.

## 1. Prepare the notebook

Use a Kaggle GPU compatible with the selected Gemma 4 checkpoint. Extract the project into `/kaggle/working/gemma4-dev-agent`, attach the model checkpoint as a Kaggle input, and follow `KAGGLE_GEMMA4_SETUP.md`.

```python
%cd /kaggle/working/gemma4-dev-agent
!python -m pytest -q
```

## 2. Run actual Gemma 4 inference

```python
MODEL_PATH = "/kaggle/input/your-gemma4-model-directory"
!cp -a examples/demo_repo /kaggle/working/demo_repo
!python scripts/run_gemma4.py \
  --model_path "$MODEL_PATH" \
  --repo_path /kaggle/working/demo_repo \
  --issue "calculator.add subtracts instead of adding. Fix it and run all tests." \
  --trajectory_path runs/trajectories/demo.json
```

The model-driven result must be captured from this command. `scripts/run_offline_demo.py` is a scripted tool-loop test and must not be described as Gemma inference.

## 3. SFT and GRPO

Follow sections 3–5 of `README.md` or the detailed commands in `KAGGLE_GEMMA4_SETUP.md`. The included example datasets only validate data formats; meaningful training requires a diverse set of successful trajectories and trusted repository snapshots.

## 4. Evaluate

Create a held-out JSONL benchmark with `issue` and `repo_path` fields, then run `scripts/evaluate_agent.py` against the base checkpoint, SFT adapter, and GRPO adapter using identical budgets. Report the generated JSON metrics. No trained model or benchmark result is bundled with this source archive.
