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
export HF_HUB_DISABLE_XET=1   # OLMo-2 fp32 checkpoints (~29 GB) can crash the hf_xet backend; use classic HTTP
```
To update code between experiments: `git pull origin main`.

**Download gotchas.** OLMo-2 checkpoints are ~29 GB each (fp32). If a download dies with
`Internal Writer Error: Background writer channel closed`, it's the `hf_xet` backend —
`export HF_HUB_DISABLE_XET=1` (or `pip uninstall -y hf_xet`) and retry; also check
`df -h /workspace` for a full disk.

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

### Pilot findings (2026-07-15)
- **Feasibility: yes.** The base model produces coherent, on-topic, persona-consistent
  generations at **n≥3, saturating ~n=5**. n=0 and n=1 are largely degenerate (repetition
  loops, verbatim demo regurgitation) — partly a greedy-decoding artifact.
- **But the base COPIES the demos, it doesn't condition on the `Context:` persona.** The
  mismatch control is decisive and symmetric: `Context:`=farmer + drill_sergeant demos → pure
  drill sergeant output; `Context:`=drill_sergeant + farmer demos → farmer output. The demos
  win in both directions; the `Context:` line is nearly inert. And n=0 (Context-only ≈ what
  base-CAA relies on) barely engages the persona — validating the original confound.
- **Implication:** few-shot-with-instruct-demos measures the *instruct teacher transferred by
  copying*, not the base's weights, so it can't serve as a clean base-intrinsic probe as-is.
  Notably, n=0 + mismatch together are *consistent with* Appendix F (base conditioning is
  absent-or-borrowed). This motivated **Experiment 3**.

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

## Experiment 3 — In-context persona generalization (base model)  🔬 IN PROGRESS

**Motivation.** The pilot (Exp 2) couldn't separate copying from conditioning because demos
and target were the *same* persona. This design fixes that: demos are **8 different (seen)
personas** answering one fixed question; the target is a **held-out** persona described only
by its system prompt. Copying can't produce a held-out persona from other personas' demos —
the base must actually read the held-out description. This separates **task-format** (from the
diverse demos) from **persona-content** (from the held-out description).

**The interesting contrast with the pilot:** there, a persona *description alone* (n=0) failed.
Here, description **+ demonstrated task format** might succeed → "the base has persona-following
machinery, but needs the task shown in-context to unlock it" (latent, elicitable). If it fails,
that's clean support for Appendix F. A positive result also yields a **copy-free** on-policy
base-generation protocol for the eventual extraction.

**Design.**
- Seen (8, in demos): farmer, drill_sergeant, con_artist, therapist, tech_ceo, kindergarten_teacher, professor, politician.
- Held-out (3, distinctive voices): surgeon, zen_master, six_year_old. Controls: null, nonsense (should read generic).
- Persona key = **full system-prompt description** (not a name — resists pretraining name-priors).
- **1 trait fixed per prompt**, run across assertiveness + honesty. Fixed question held constant so persona is the only variable.
- **n-sweep** over the number of seen-persona demos, n ∈ {1,3,5,7} (nested) — measures how many diverse examples the base needs before it keys on the held-out *description* rather than the nearest demo.
- Controls (inline): **swap-description** (held-out personas share the same demo block per n), **n=0** (description only), **shuffle** (reversed demo order at max n = recency control), **null/nonsense** targets.
- Decoding: sampling + `repetition_penalty` + `no_repeat_ngram_size` (avoids the pilot's greedy degeneration).

**Run:**
```bash
bash scripts/run_persona_generalization.sh
#   = f1 (instruct demos, 11 personas) -> f3 (base generates held-out personas)
python pipeline/f3_persona_generalization.py --dry-run   # preview a prompt without loading models
```
**Outputs:** `results/persona_generalization/generations.md` (grouped by trait×question, each
held-out persona's n-sweep n=0/1/3/5/7 + shuffle next to the instruct reference) +
`base_generations.jsonl`.

**What to look for.** For a fixed (trait, question): do the held-out personas produce *distinct*
voices matching their descriptions (surgeon clinical, zen_master koan-like, six_year_old
childish)? **At what n does the held-out voice emerge / stabilize** (n-sweep sample complexity)?
Do null/nonsense read generic? Does any n beat `n=0` (task-format helps beyond a bare
description)? Is output stable under `shuffle` (not just copying the last demo)? If yes across
the board → the base conditions on a persona description when the task is demonstrated. Next:
quantify with the persona classifier / LLM judge, then decide on extraction — keeping the live
confounds (contamination, behavior≠geometry, not-CAA-comparable) in view.

## Pipeline reference (files this sprint touches)

| Step | Script | Purpose |
|---|---|---|
| t1–t4 | `pipeline/t1_trajectory_activations.py` … `t4_trajectory_figures.py` | OLMo trajectory (CAA across checkpoints) |
| — | `scripts/run_olmo_trajectory.sh` | one-shot trajectory runner + result collection |
| — | `scripts/sweep_trajectory_layer.py` | pick the analysis layer matching the paper |
| — | `scripts/summarize_trajectory.py` | write `results/olmo_trajectory/SUMMARY.md` |
| — | `scripts/variance_by_persona.py` | Experiment 1b per-persona plot |
| f1 | `pipeline/f1_fewshot_demos.py` | instruct-model demos (persona+trait) for the pilots |
| f2 | `pipeline/f2_fewshot_base_generate.py` | Exp 2: few-shot the base model (same-persona demos) |
| f3 | `pipeline/f3_persona_generalization.py` | Exp 3: base generates HELD-OUT personas from seen-persona demos |
| — | `scripts/run_fewshot_pilot.sh` | Exp 2 runner (f1 → f2) |
| — | `scripts/run_persona_generalization.sh` | Exp 3 runner (f1 → f3) |
