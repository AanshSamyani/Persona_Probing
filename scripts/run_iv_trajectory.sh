#!/usr/bin/env bash
# IV variant of the OLMo-2 training-trajectory experiment.
#   step 0: generate the fixed SFT demo pool (pos+neg) for all personas x traits
#   tiv1:   few-shot each checkpoint with those demos, extract mean-over-generated activations
#   tiv2:   contrastive IV vectors mean(pos)-mean(neg)
#   tiv3:   trajectory analysis (shared-variance ρ, transfer matrices)
#   tiv4:   figures + SUMMARY (skeptical traits flagged)
#
# All local model paths under /workspace/models (no HF cache). GPU required.
#   bash scripts/run_iv_trajectory.sh
#   LAYER=15 bash scripts/run_iv_trajectory.sh
set -euo pipefail
source /workspace/env.sh 2>/dev/null || true
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
LAYER="${LAYER:-15}"

CORE="farmer politician therapist drill_sergeant street_hustler professor tech_ceo kindergarten_teacher surgeon con_artist"
DEMOS=outputs/OLMo-2-1124-7B-SFT/iv_demos

echo "=== step 0: SFT demo pool (all personas x 8 traits, pos+neg) ==="
python pipeline/f1_fewshot_demos.py \
  --model /workspace/models/OLMo-2-1124-7B-SFT \
  --personas $CORE \
  --traits assertiveness confidence deference empathy honesty impulsivity risk_taking warmth \
  --n-demo-questions 10 \
  --output-dir "$DEMOS"

echo "=== tiv1: few-shot IV extraction across all 7 checkpoints ==="
python pipeline/tiv1_extract.py --demos-dir "$DEMOS"

echo "=== tiv2: contrastive IV vectors ==="
python pipeline/tiv2_vectors.py

echo "=== tiv3: trajectory analysis (layer $LAYER) ==="
python pipeline/tiv3_analysis.py --layer "$LAYER"

echo "=== tiv4: figures + SUMMARY ==="
python pipeline/tiv4_figures.py

echo ""
echo "=== DONE. results/iv_trajectory/  (SUMMARY.md + fig_iv_variance_trajectory.png) ==="
