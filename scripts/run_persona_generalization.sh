#!/usr/bin/env bash
# Experiment 3: in-context persona generalization (base model).
#   f1: on-policy demos from OLMo-2-Instruct for 8 seen + 3 held-out personas
#   f3: few-shot the base model with the 8 seen personas, ask it to generate a HELD-OUT persona
#
# Same transformers env as the trajectory/pilot (no vLLM). GPU required.
# Base weights: /workspace/models/OLMo-2-1124-7B if present, else the Hub id.
#
#   bash scripts/run_persona_generalization.sh
set -euo pipefail

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

DEMOS_DIR="outputs/OLMo-2-1124-7B-Instruct/persona_gen/demos"

echo "=== f1: on-policy demos from OLMo-2-Instruct (8 seen + 3 held-out personas) ==="
python pipeline/f1_fewshot_demos.py \
  --personas farmer drill_sergeant con_artist therapist tech_ceo kindergarten_teacher professor politician \
             surgeon zen_master six_year_old \
  --traits assertiveness honesty \
  --output-dir "$DEMOS_DIR"

echo "=== f3: base model generates HELD-OUT personas from seen-persona demos ==="
python pipeline/f3_persona_generalization.py --demos-dir "$DEMOS_DIR"

echo ""
echo "=== DONE. Eyeball: results/persona_generalization/generations.md ==="
echo "    Raw: results/persona_generalization/base_generations.jsonl"
