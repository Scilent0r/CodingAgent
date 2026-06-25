# Local Coding Agent (ReAct + Test Loop)

A minimal prototype: a local LLM (served by KoboldCPP) writes code, runs pytest,
reads the failures, and fixes its own code in a loop until tests pass.

## 1. Model

Recommended: **Qwen3-Coder-30B-A3B-Instruct**, GGUF, Q4_K_M or Q8_0 quant.
Download from Hugging Face (search "Qwen3-Coder-30B-A3B-Instruct GGUF", e.g. the
Unsloth or bartowski repacks). Q4_K_M is ~18-20GB, Q8_0 is ~32GB — Q4_K_M leaves
more headroom for context on a 32GB card.

## 2. Launch KoboldCPP

Example launch (adjust paths):

```
koboldcpp.exe --model "C:\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" ^
  --contextsize 16384 ^
  --gpulayers 999 ^
  --flashattention ^
  --usecublas ^
  --jinja --jinjatools ^
  --port 5001
```

**`--jinja --jinjatools` is required** as of the native tool-calling overhaul. This
tells KoboldCPP to use the model's own chat template — which for Qwen3 includes
proper Hermes-style tool-call support — instead of KoboldCPP's generic "universal"
tool-call heuristics, which are not what this project's prompts are built around.

Honest caveat: KoboldCPP's own docs list Gemma3, not Qwen, as their recommended
tool-calling model, and native tool calling there is a fairly recent, still-evolving
feature. It should work well with Qwen3-Coder (which was trained on Hermes-style tool
use specifically), but if you hit weird tool-call failures, that's the first thing to
suspect — check you're on a recent KoboldCPP build and that both flags are actually
present in your launch command.

- `--gpulayers 999` offloads every layer to the GPU (it'll clamp to however many
  actually fit).
- Confirm the server is up: open `http://localhost:5001/v1/models` in a browser,
  should return JSON.

## 3. Python environments (Anaconda)

Two envs, kept deliberately separate:

```bash
# Main env — runs the orchestrator
conda create -n coding-agent python=3.11 -y
conda activate coding-agent
pip install -r requirements.txt

# Sandbox env — ONLY what generated code is allowed to import.
# Keep this minimal on purpose.
conda create -n agent-sandbox python=3.11 pytest -y
```

After creating `agent-sandbox`, find its python.exe path:

```bash
conda env list --json
```

Copy that path into `config.py` (created on first run) as `SANDBOX_PYTHON`.

## 4. Run the prototype

```bash
conda activate coding-agent

python run_task.py --task example_tasks/fizzbuzz.txt
python run_task.py --task example_tasks/pricing_bug.txt --seed example_tasks/seed_pricing_bug --protect test_pricing.py
python run_task.py --task example_tasks/ambiguous_name_format.txt
python run_task.py --task example_tasks/node_bug.txt --seed example_tasks/seed_node_bug --protect stringUtils.test.js --language node
python run_task.py --task example_tasks/memory_demo_part1.txt
python run_task.py --task example_tasks/memory_demo_part2.txt
python run_task.py --task example_tasks/duration_lookup.txt
python run_task.py --task example_tasks/web_search_smoke_test.txt
```

Watch the console — it prints each thought/action/observation as the loop runs.

## How it works

1. `orchestrator.py` keeps a chat history and asks the model for the *next single
   action* as strict JSON (enforced via KoboldCPP's structured-output / grammar
   support — see `llm_client.py`).
2. The action is one of: `write_file`, `read_file`, `run_tests`, `finish`.
3. Every action's result (file written / test output) is appended to history as
   an observation, and the loop continues.
4. `finish` is only accepted if the most recent test run passed. Otherwise the
   model is told "tests are not passing yet, keep going."
5. Everything happens inside `./runs/<task_id>/` — a clean directory per task.

This is intentionally bare-bones — no Docker, no network sandboxing yet. Good
enough for self-contained function+test problems. We'll harden it once the loop
itself is solid.
