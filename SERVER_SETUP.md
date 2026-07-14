# Server setup — replicate the OLMo-2 training trajectory (Appendix F, Fig 19)

Everything lives under `/workspace` (the only persistent path). Weights, HF cache,
activations and vectors are gitignored, so pushing results back never drags them
onto another machine. Only code, data, and `results/` travel through GitHub.

## Prerequisites

- 1 GPU with ≥ 24 GB (OLMo-2-7B is ~14 GB in bf16; forward-pass only).
- **~150 GB free on `/workspace`** — 7 checkpoints (~14 GB each) + activations.
  Disk-constrained? See "Stage-by-stage" at the bottom.
- Python 3.10+.

## 0. Clone into /workspace

```bash
cd /workspace
git clone https://github.com/AanshSamyani/Persona_Probing.git
cd Persona_Probing
```

## 1. Virtualenv + dependencies

```bash
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip

# assistant_axis provides ProbingModel (activation extraction); persona_steering is this repo.
git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref
pip install -e assistant-axis-ref/
pip install -e .

# OLMo-2 needs a recent transformers — install LAST so nothing downgrades it.
pip install -U "transformers>=4.47"
```

Verify torch sees the GPU (if `False`, install the CUDA build matching your driver):

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import assistant_axis, persona_steering; print('imports ok')"
```

## 2. Environment (caches on /workspace, W&B off)

```bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
# OLMo-2 is open (no token needed). Only if a checkpoint is gated:
# export HF_TOKEN=hf_xxx
```

The CAA A/B question sets are already committed under `data/prompts/caa/` and the
persona YAMLs under `data/personas/`, so **no data-generation / API key is needed**.

## 3. Run the trajectory pipeline

One command runs t1→t4 and collects results:

```bash
bash scripts/run_olmo_trajectory.sh          # analysis layer 15 (default)
```

Or step-by-step (same thing, so you can watch each stage):

```bash
python pipeline/t1_trajectory_activations.py            # extract CAA acts, all 7 stages (GPU, slow: downloads weights)
python pipeline/t2_trajectory_vectors.py                # mean(pos)-mean(neg) per persona x trait, all layers
python scripts/sweep_trajectory_layer.py                # pick the layer whose pretrain≈0.96 / post≈0.85
python pipeline/t3_trajectory_analysis.py --layer 15    # transfer matrices + shared variance
python pipeline/t4_trajectory_figures.py                # figures
python scripts/summarize_trajectory.py                  # writes results/olmo_trajectory/SUMMARY.md
```

## 4. Inspect

```bash
cat results/olmo_trajectory/SUMMARY.md
ls  results/olmo_trajectory/figures/     # fig_variance_trajectory.png == paper Figure 19
```

Expected (paper targets): shared variance ≈ 95.7 / 96.5 / 96.0 / 96.8 across
pretrain 1/10/50% → base, then a ~11-point drop to ≈ 85.5 / 84.2 / 83.5 at
SFT / DPO / Instruct.

## 5. Push results back to GitHub

Only `results/` (+ any code tweaks) are tracked; weights/activations are gitignored.

```bash
git add results/ && git status          # confirm: no *.pt / *.safetensors / outputs/ staged
git commit -m "OLMo-2 trajectory results (Fig 19 replication)"
git push origin main
```

## Stage-by-stage (disk-constrained)

`t1` accepts `--stages`. Extract one checkpoint, then free its weights before the
next (activations are tiny and are what you keep):

```bash
for S in pretrain_1pct pretrain_10pct pretrain_50pct base sft dpo instruct; do
  python pipeline/t1_trajectory_activations.py --stages "$S"
  python pipeline/t2_trajectory_vectors.py     --stages "$S"
  rm -rf "$HF_HOME/hub"/models--allenai--OLMo-2-1124-7B*   # frees ~14 GB; re-downloads only if re-run
done
python pipeline/t3_trajectory_analysis.py --layer 15
python pipeline/t4_trajectory_figures.py
python scripts/summarize_trajectory.py
```
