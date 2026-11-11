# AST-Guided Self-Correcting Coding Agent via SFT + GRPO on Gemma 4

**Paper Track Submission — Gemma 4 Developer Agent Competition (Kaggle, 2026)**

## Abstract

We describe an implementation and evaluation plan for post-training an open Gemma 4 model as an offline Python coding agent. The current release contains the agent framework, AST tools, trajectory pipeline, LoRA SFT script, execution-reward group-relative policy optimization script, and evaluation harness. Model-level reliability improvements remain hypotheses until the planned GPU experiments are completed. The agent combines (1) deterministic Abstract Syntax Tree (AST) tools for precise symbol location and safe editing, (2) supervised fine-tuning (SFT) on code-editing trajectories, and (3) Group Relative Policy Optimization (GRPO) whose reward is derived from local pytest execution. The resulting system can search, patch, run tests, observe failures, and self-correct inside a fixed time and tool-call budget. We provide code intended to support reproduction on compatible Kaggle GPU hardware; exact checkpoint compatibility, completed training, and measured improvements must be verified experimentally.

## 1. Introduction

Modern coding agents achieve impressive results when backed by large proprietary models and unrestricted tool access. However, many developers need an agent that can run offline and respect strict time and tool-call budgets. The Gemma 4 Developer Agent competition formalizes this setting: a quantized Gemma 4 model must navigate real repositories, produce patches, and pass hidden validation tests without internet access.

We focus on three complementary levers:

- **Deterministic navigation.** Large language models frequently hallucinate file contents or line numbers. We therefore expose an AST-based `search_symbol` tool that returns exact definitions, reducing the need for the model to memorize the entire codebase.
- **Trajectory SFT.** We collect and synthesize editing trajectories (issue → tool calls → final patch) and fine-tune Gemma 4 with PEFT LoRA.
- **Execution-based GRPO.** After SFT we further optimize the policy with Group Relative Policy Optimization, using the binary (or shaped) outcome of local pytest runs as the reward signal. This teaches the model to value patches that actually make tests pass and to recover from intermediate failures.

## 2. Method

### 2.1 Agent Tools

The agent exposes six deterministic repository tools implemented in Python:

- `repository_map()` – indexes Python files, AST definitions, imports, and references.
- `search_symbol(name)` – walks the AST of Python files and returns matching function/class definitions with source snippets.
- `read_file(path)` – returns file content (truncated for long files).
- `write_file(path, content)` – preferred editing primitive for focused changes.
- `syntax_check(path)` – `compile(..., mode="exec")` gate before testing.
- `run_pytest(...)` – executes a targeted pytest invocation with a hard timeout and returns a compressed result.

These tools are deterministic, side-effect free (except write), and require no network.

### 2.2 Training Pipeline

**Stage 1 – SFT.**  
We fine-tune a selected Gemma 4 checkpoint using PEFT LoRA on successful tool-agent trajectories converted into direct patch-completion examples. The script supports configurable sequence length, LoRA rank, epochs, and batch accumulation. The dataset should be built from successful runs on training repositories only; no benchmark contamination is permitted. No completed SFT run is claimed in this draft.

**Stage 2 – GRPO.**  
Starting from the SFT adapter, the supplied custom GRPO-style trainer samples a group of patch completions for each issue. Each candidate is applied to a fresh repository snapshot and evaluated with pytest. Rewards distinguish malformed patches, syntax failures, test failures, and successful test runs. Group-normalized advantages and a clipped sequence-level policy objective update the adapter. This is a compact research implementation; its convergence and stability must be measured, and it should not be presented as equivalent to every feature of a production GRPO implementation.

### 2.3 Self-Correction Loop

At inference the controller maintains a message history and is instructed to:

1. Locate the relevant code with `search_symbol` or `read_file`,
2. Propose a minimal edit via `write_file`,
3. Verify syntax,
4. Run tests,
5. If tests fail, use the observed output to revise; report success only after a full unfiltered pytest run passes after the latest edit.

The loop is terminated by a success message, a maximum turn count, or exhaustion of the time/tool budget.

## 3. Experiments

The planned evaluation uses a held-out subset of public development tasks (for example, fastapi, rich, requests, and httpx) and a separate suite of synthetic bugs. Metrics to report are:

- Resolution rate (fraction of tasks whose tests pass after the agent finishes),
- Average tool calls and wall-clock time per solved task,
- Ablation of AST tools versus pure file reading,
- Ablation of GRPO versus SFT-only.

(No benchmark results are included in this code release. Run the supplied evaluation harness on a held-out benchmark and report measured numbers before making comparative performance claims.)

## 4. Related Work

Our work sits at the intersection of tool-augmented language models, parameter-efficient fine-tuning for code, and reinforcement learning from execution feedback. We build on ideas from SWE-bench, CodeAct, OpenDevin, and recent GRPO applications, while deliberately constraining the setting to a single quantized open model and fully offline operation.

## 5. Conclusion

The implementation provides a concrete framework for testing whether deterministic AST tools, trajectory SFT, and execution-rewarded group-relative optimization improve Gemma 4 on offline coding tasks. Whether they materially improve reliability is an empirical question for the completed experiments.

## Limitations

- GRPO sample efficiency remains low; each reward evaluation requires a full agent roll-out.
- The current patch application path uses complete-file rewrites rather than applying general unified diffs.
- Evaluation is still limited to the public development distribution; generalization to completely unseen repositories needs further study.

## Reproducibility

The code, training scripts, sample data, and reproducibility notes are included in this project archive. The scripted offline demonstration exercises the search → edit → fail → self-correct → pass tool sequence; it is not Gemma 4 inference. A genuine model-driven recording should be added only after a real checkpoint run.
