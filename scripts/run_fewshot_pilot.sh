#!/usr/bin/env bash
# Few-shot ICL base-extraction pilot (design B, feasibility only).
#   f1: generate persona+trait demos from OLMo-2-Instruct
#   f2: few-shot the OLMo-2 BASE model with n in {0,1,3,5,7} demos, dump generations
#
# Runs in the same transformers env as the trajectory (no vLLM). GPU required.
# base + instruct weights are already in the HF cache from the trajectory run
# (unless you purged them with the stage-by-stage cleanup — then they re-download).
#
#   bash scripts/run_fewshot_pilot.sh
set -euo pipefail

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# OLMo-2 fp32 checkpoints can crash the hf_xet download backend; use classic HTTP. Override with =0.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "=== f1: instruct-model demos (OLMo-2-1124-7B-Instruct) ==="
python pipeline/f1_fewshot_demos.py

echo "=== f2: few-shot the base model (OLMo-2-1124-7B) + dump generations ==="
python pipeline/f2_fewshot_base_generate.py --directions pos

echo ""
echo "=== DONE. Eyeball: results/fewshot_pilot/generations.md ==="
echo "    Raw generations: results/fewshot_pilot/base_generations.jsonl"
echo "    Demos (instruct, gitignored): outputs/OLMo-2-1124-7B-Instruct/fewshot_pilot/demos/"
