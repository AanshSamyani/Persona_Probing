# Sprint Week Experiments

Living log of the experiments we're running on top of
[`jacobdaviescam/steering_across_personas`](https://github.com/jacobdaviescam/steering_across_personas)
to replicate and extend the persona-conditioning / trait-geometry results from the
*Personas Shape How Models Represent Behaviors* paper on **OLMo-2**.

**Conventions**
- All runs happen on the SSH server under `/workspace` (only persistent path). Code is
  written/understood locally and pushed to `github.com/AanshSamyani/Persona_Probing`.
- Heavy artifacts (weights, activations, vectors, HF cache) stay under `outputs/` and the
  HF cache — **gitignored, never pushed**. Only code, data, and `results/` (numbers + plots
  + generations) go to GitHub.
- Env: Python venv at `/workspace/venv`, `transformers>=4.47` (OLMo-2 support),
  `WANDB_DISABLED=true`, HF cache at `/workspace/.cache/huggingface`. Full setup in
  [`SERVER_SETUP.md`](SERVER_SETUP.md).

---

## One-time setup

See [`SERVER_SETUP.md`](SERVER_SETUP.md). Summary:
```bash
cd /workspace && git clone https://github.com/AanshSamyani/Persona_Probing.git && cd Persona_Probing
python -m venv /workspace/venv && source /workspace/venv/bin/activate && pip install --upgrade pip
git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref
pip install -e assistant-axis-ref/ && pip install -e . && pip install -U "transformers>=4.47"
export HF_HOME=/workspace/.cache/huggingface HUGGINGFACE_HUB_CACHE=$HF_HOME/hub WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
```
To update code between experiments: `git pull origin main`.

---

## Experiment 1 — OLMo-2 training-trajectory replication (paper Appendix F, Fig 19)  ✅ DONE

**Question.** When in training does a model start representing behavioral traits differently
per persona? **Method.** For 7 OLMo-2 checkpoints (pretrain 1/10/50% → base → SFT → DPO →
Instruct), extract per-persona trait vectors with **CAA** (single forward pass, activation
at the answer-letter token — works on base models, which can't be system-prompted or follow
instructions), then measure the **shared variance ratio** ρ: the fraction of each trait's
cross-persona vector energy lying along the shared consensus direction. ρ≈1 = trait is
persona-universal; lower ρ = persona-conditioned.

**Result (reproduced exactly).** ρ ≈ 0.96 flat through all pretraining + base, then a sharp
~11-point drop at **SFT** (→ ~0.85), small further decline through DPO/Instruct ⇒
**persona-conditional trait geometry is created by post-training, not pretraining.**

**Run:**
```bash
bash scripts/run_olmo_trajectory.sh          # t1 extract → t2 vectors → t3 analysis (layer 15) → t4 figures
# step-by-step + layer sweep + disk-constrained variant: see SERVER_SETUP.md
```
**Outputs:** `results/olmo_trajectory/` — `SUMMARY.md` (Fig 19 + Table 6 tables),
`variance_trajectory.json`, `figures/fig_variance_trajectory.png` (= Figure 19).

---

## Experiment 1b — Per-persona shared-variance companion plot  ✅ DONE

**Question.** Figure 19 shows one line per *trait* (averaged over personas). Which *personas*
drive the SFT drop? **Method.** Same building block `cos²(v_{c,T}, ŝ_T)` (persona c's trait-T
vector vs the trait's consensus direction), but averaged **over traits per persona** instead
of over personas per trait — the other marginal of the same matrix. Consistency check: the
‖v‖²-weighted mean over personas reconstructs the aggregate ρ from Experiment 1.

**Run** (reads existing t2 vectors + `trajectory_meta.json`; no re-extraction):
```bash
python scripts/variance_by_persona.py
```
**Outputs:** `results/olmo_trajectory/figures/fig_variance_by_persona.png`,
`variance_by_persona.json`.

---

## Experiment 2 — Few-shot ICL base-extraction pilot (design B)  🔬 IN PROGRESS (feasibility)

**Motivation — the confound in Experiment 1's base extraction.** On the base model, CAA reads
the answer-letter activation after a raw `Context: {persona}` prefix. The risk isn't noise,
it's **inertness**: ρ≈0.96 at base could mean the trait really is persona-universal, OR that
the base model **barely conditions on the raw-text persona at all**, collapsing every persona
onto the same direction (high ρ by default, not by structure). The paper half-admits this
(ρ=0.96 ≠ 1.0). Few-shot demonstrations are much harder to ignore than a `Context:` sentence,
so they test whether persona-conditioning appears once the persona has real purchase.

**Idea (design B).** Give the base model `n` in-context **demonstrations** of a persona
expressing a trait — generated **on-policy by the instruct model** — then let the base model
generate on a held-out question. ICL is something base models *can* do (unlike instruction-
following, which is why IV extraction is impossible on base).

**This pilot = feasibility only.** We eyeball whether base generations get more coherent,
more in-persona, and more trait-expressing as **n ∈ {0,1,3,5,7}** grows. **No activation
extraction or vectors yet.** Persona is conveyed via demos **+** an explicit `Context:` line.

**Design.**
- Cells: personas × traits = {`drill_sergeant`, `con_artist`, `farmer`} × {`assertiveness`, `honesty`}.
- Demo source: `OLMo-2-1124-7B-Instruct` (the "final" model). *(No OLMo IV generations existed —
  the paper's IV responses are Gemma-only and the trajectory was CAA/forward-pass — so these are
  generated fresh, which is exactly the intended on-policy-from-instruct source.)*
- Prompt to base: `Context: {persona}` + n demos (`Question:/Answer:` from instruct) + held-out `Question:/Answer:`.
- Controls baked in: **n=0** baseline, and **mismatched-persona demos** (same `Context:`, demos
  from a different persona — if the answer follows the demo persona, it's copying, not conditioning).
- Backend: HuggingFace `transformers.generate()` (no vLLM), same env as the trajectory.

**Run:**
```bash
bash scripts/run_fewshot_pilot.sh
#   = python pipeline/f1_fewshot_demos.py           # instruct demos → outputs/.../fewshot_pilot/demos/ (gitignored)
#     python pipeline/f2_fewshot_base_generate.py   # base few-shot gens → results/fewshot_pilot/
# preview without loading models:
python pipeline/f1_fewshot_demos.py --dry-run
python pipeline/f2_fewshot_base_generate.py --dry-run
```
**Outputs:** `results/fewshot_pilot/generations.md` (grouped, scannable) and
`base_generations.jsonl`.

**What to look for.** (1) Coherence — does the base stop rambling and answer the question?
(2) Persona voice — does it adopt the persona as n grows? (3) Trait expression — is the
answer honest/assertive as intended? (4) Copying vs conditioning — does `mismatch` track the
Context persona (good) or the demo persona (copying)? The n at which quality saturates tells
us a sensible operating point for the full experiment.

### Planned follow-ups (the full design-B experiment — not yet run)
Only pursue if the pilot shows the base model can do the task:
1. **Extract** mean assistant-turn activations from the base few-shot generations (IV-style),
   build pos/neg contrastive trait vectors (pos demos vs neg demos, persona held fixed).
2. **Re-measure the ρ trajectory** across all 7 stages under the *same* few-shot method — does
   the SFT cliff survive a base-fair, naturalistic extraction? (survives → result is robust;
   flattens → the original drop was partly a measurement artifact).
3. **Demo-source ablation** (the key contamination control): instruct vs SFT vs DPO vs
   base-self-generated vs human-written demos. If conditioning tracks the source → it's the
   demos, not the base weights.
4. **n-scaling as a diagnostic:** monotone rise with n ⇒ in-context injection; early saturation
   ⇒ the base recognizes the persona and applies its own conditioning.
5. **Interpretation ceiling:** few-shot changes *context*, not *weights*. Even a clean positive
   refines "post-training *creates* conditioning" to "the base has latent, ICL-elicitable
   conditioning that post-training makes weight-default" — a nuance, not a refutation.

---

## Pipeline reference (files this sprint touches)

| Step | Script | Purpose |
|---|---|---|
| t1–t4 | `pipeline/t1_trajectory_activations.py` … `t4_trajectory_figures.py` | OLMo trajectory (CAA across checkpoints) |
| — | `scripts/run_olmo_trajectory.sh` | one-shot trajectory runner + result collection |
| — | `scripts/sweep_trajectory_layer.py` | pick the analysis layer matching the paper |
| — | `scripts/summarize_trajectory.py` | write `results/olmo_trajectory/SUMMARY.md` |
| — | `scripts/variance_by_persona.py` | Experiment 1b per-persona plot |
| f1 | `pipeline/f1_fewshot_demos.py` | instruct-model demos for the few-shot pilot |
| f2 | `pipeline/f2_fewshot_base_generate.py` | few-shot the base model + dump generations |
| — | `scripts/run_fewshot_pilot.sh` | one-shot pilot runner (f1 → f2) |
