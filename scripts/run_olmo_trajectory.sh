#!/usr/bin/env bash
# Replicate the OLMo-2 training-trajectory result (paper Appendix F: Figure 19 + Table 6).
#
# Run from the repo root on the server, inside the venv, with assistant-axis-ref
# installed (see SERVER_SETUP.md). GPU required for step t1 only.
#
#   bash scripts/run_olmo_trajectory.sh              # analysis layer 15 (default)
#   LAYER=14 bash scripts/run_olmo_trajectory.sh     # override analysis layer
#
# Idempotent: t1/t2 skip files that already exist, so you can re-run safely.
set -euo pipefail

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
LAYER="${LAYER:-15}"

MODEL_SHORT="OLMo-2-1124-7B"
TDIR="outputs/${MODEL_SHORT}/trajectory"
FDIR="outputs/${MODEL_SHORT}/figures/trajectory"

echo "=== [t1] extract CAA activations across OLMo checkpoints (GPU, downloads ~7 checkpoints) ==="
python pipeline/t1_trajectory_activations.py

echo "=== [t2] contrastive vectors per stage (all layers stored) ==="
python pipeline/t2_trajectory_vectors.py

echo "=== [sweep] mean shared variance per candidate layer (informational; pick the layer matching the paper) ==="
python scripts/sweep_trajectory_layer.py || true

echo "=== [t3] cross-stage analysis at layer ${LAYER} ==="
python pipeline/t3_trajectory_analysis.py --layer "${LAYER}"

echo "=== [t4] figures (fig_variance_trajectory.png == paper Figure 19) ==="
python pipeline/t4_trajectory_figures.py

echo "=== [collect] copy shareable results into results/olmo_trajectory/ ==="
mkdir -p results/olmo_trajectory/figures
cp -f "${TDIR}"/*.json results/olmo_trajectory/         2>/dev/null || true
cp -f "${TDIR}"/*.npy  results/olmo_trajectory/         2>/dev/null || true
cp -f "${FDIR}"/*.png  results/olmo_trajectory/figures/ 2>/dev/null || true
python scripts/summarize_trajectory.py

echo "=== [extra] per-persona shared-variance plot (personas as lines, averaged over traits) ==="
python scripts/variance_by_persona.py

echo ""
echo "=== DONE. Shareable results are in results/olmo_trajectory/ (safe to git add / commit / push). ==="
echo "    Heavy artifacts stayed under outputs/ and $HF_HOME (gitignored)."
