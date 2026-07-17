#!/usr/bin/env python3
"""Layer selection, criterion (i): self-steering behavioral-lift sweep  (paper §A.2).

Reproduces the paper's primary layer-selection criterion: for each candidate
layer, steer the persona-prompted model with the cell's OWN trait vector
(self-steer, alpha=2, L2-normalized per §A.3), generate, and score the trait with
a held-out Claude judge. Lift = mean(self-steered trait score) - mean(baseline
trait score). The paper picks a layer whose lift is "among the largest in the
sweep" AND whose bootstrap stability (ls_bootstrap_sweep.py) is high.

Design: loads the model ONCE, caches each cell's baseline (layer-independent), and
sweeps all layers internally. Steering uses the same assistant_axis.ActivationSteering
path as pipeline/8_steered_generation.py, so the alpha/normalize semantics match the
paper exactly. Judged via OpenRouterJudge (repo's 9_steering_eval hardcodes the
Anthropic API, which we don't use).

Cost: judge calls = n_cells * n_questions * (1 baseline + n_layers). Use --dry-run
to print the count and rough $, and start with a cell subset + --judge-model haiku.

Requires OPENROUTER_API_KEY. Runs on the GPU server (needs the model + assistant-axis-ref).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python pipeline/ls_selfsteer_sweep.py \
      --model /workspace/models/Qwen2.5-7B-Instruct \
      --vectors-dir outputs/Qwen2.5-7B-Instruct/caa_vectors \
      --layers 10 14 16 18 19 20 21 22 24 26 \
      --alpha 2 --n-questions 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "assistant-axis-ref"))

from persona_steering.config import Trait
from persona_steering.data import load_all_trait_datasets
from persona_steering.personas import load_all_personas
from persona_steering.openrouter_judge import OpenRouterJudge
from persona_steering.utils import log

# cost-conscious defaults (subset). Widen with --personas/--traits ... or --all-cells.
DEF_PERSONAS = ["farmer", "drill_sergeant", "therapist", "con_artist"]
DEF_TRAITS = ["assertiveness", "empathy", "honesty", "confidence"]
COST_PER_CALL = {"anthropic/claude-sonnet-4.5": 0.0015, "anthropic/claude-haiku-4.5": 0.0004}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-steer behavioral-lift layer sweep (criterion i)")
    p.add_argument("--model", required=True, help="HF id or local path (Qwen2.5-7B-Instruct)")
    p.add_argument("--vectors-dir", required=True, help="CAA vectors dir ({persona}_{trait}.pt)")
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--alpha", type=float, default=2.0, help="steering coefficient (paper: 2)")
    p.add_argument("--no-normalize", action="store_true",
                   help="disable L2-normalize+mean-rescale (paper uses normalized vectors)")
    p.add_argument("--personas", nargs="+", default=DEF_PERSONAS)
    p.add_argument("--traits", nargs="+", default=DEF_TRAITS)
    p.add_argument("--all-cells", action="store_true", help="use all 10 personas x 8 traits")
    p.add_argument("--n-questions", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-4.5",
                   help="OpenRouter slug; paper judge is sonnet-4.5, haiku-4.5 is ~4x cheaper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="print gen+judge counts and cost, no model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "layer_select" / "selfsteer"
    vdir = Path(args.vectors_dir)

    all_personas = load_all_personas()
    pmap = {p.slug: p for p in all_personas}
    if args.all_cells:
        personas = [p for p in ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
                                "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
                    if p in pmap]
        traits = list(Trait)
    else:
        personas = args.personas
        traits = [Trait(t) for t in args.traits]

    # sample the same questions per trait (shared across cells/layers)
    datasets = load_all_trait_datasets(traits)
    rng = random.Random(args.seed)
    qs_by_trait = {}
    for t in traits:
        ds = datasets.get(t)
        if ds is None:
            continue
        qs = list(ds.questions)
        qs_by_trait[t.value] = rng.sample(qs, args.n_questions) if args.n_questions < len(qs) else qs

    cells = [(p, t) for p in personas for t in traits if t.value in qs_by_trait
             and (vdir / f"{p}_{t.value}.pt").exists()]
    n_q = args.n_questions
    judge_calls = len(cells) * n_q * (1 + len(args.layers))
    est = COST_PER_CALL.get(args.judge_model, 0.0015) * judge_calls

    print(f"Layers:   {args.layers}")
    print(f"Cells:    {len(cells)} ({len(personas)} personas x {len(traits)} traits, vectors present)")
    print(f"Questions/cell: {n_q}   alpha={args.alpha}  normalize={not args.no_normalize}")
    print(f"Generations: baseline {len(cells)*n_q} + steered {len(cells)*len(args.layers)*n_q}")
    print(f"Judge model: {args.judge_model}")
    print(f"Judge calls: {judge_calls}   rough cost: ~${est:.2f}")
    if args.dry_run:
        return

    # ---- load model + judge ----
    from assistant_axis.internals.model import ProbingModel
    from assistant_axis.steering import ActivationSteering
    from assistant_axis.generation import generate_response, format_conversation

    log.info("loading model %s ...", args.model)
    probing = ProbingModel(args.model)
    model, tok = probing.model, probing.tokenizer
    judge = OpenRouterJudge(model=args.judge_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_f = open(out_dir / "generations.jsonl", "a")

    def gen(system_prompt, question, steer=None, layer=None):
        conv = format_conversation(system_prompt, question, tok)
        if steer is None:
            return generate_response(model, tok, conv, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature)
        with ActivationSteering(model, steering_vectors=[steer], coefficients=[args.alpha],
                                layer_indices=[layer]):
            return generate_response(model, tok, conv, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature)

    # precompute per-layer normalized self-steer vectors (paper §A.3: L2-normalize, rescale to mean ||v||)
    def steer_vecs_at(L):
        raw = {}
        for p, t in cells:
            fv = torch.load(vdir / f"{p}_{t.value}.pt", map_location="cpu", weights_only=False)["vector"]
            if L < fv.shape[0]:
                raw[(p, t.value)] = fv[L].float()
        if not args.no_normalize and raw:
            mean_norm = float(torch.stack([v.norm() for v in raw.values()]).mean())
            raw = {k: v / (v.norm() + 1e-8) * mean_norm for k, v in raw.items()}
        return raw

    # ---- baselines (once per cell) ----
    baseline = {}
    for p, t in cells:
        sp = pmap[p].default_system_prompt
        sc = []
        for q in qs_by_trait[t.value]:
            r = gen(sp, q)
            s = judge.score_trait(r, t).score
            sc.append(s)
            gen_f.write(json.dumps({"phase": "baseline", "persona": p, "trait": t.value,
                                    "question": q, "response": r, "trait_score": s}) + "\n")
        baseline[(p, t.value)] = float(np.mean(sc))
        log.info("baseline %s/%s = %.3f", p, t.value, baseline[(p, t.value)])

    # ---- steered, per layer ----
    per_layer = {}
    for L in args.layers:
        sv = steer_vecs_at(L)
        lifts, steered_means = [], []
        for p, t in cells:
            key = (p, t.value)
            if key not in sv:
                continue
            sp = pmap[p].default_system_prompt
            sc = []
            for q in qs_by_trait[t.value]:
                r = gen(sp, q, steer=sv[key].to(model.device), layer=L)
                s = judge.score_trait(r, t).score
                sc.append(s)
                gen_f.write(json.dumps({"phase": "steered", "layer": L, "persona": p, "trait": t.value,
                                        "question": q, "response": r, "trait_score": s}) + "\n")
            steered_mean = float(np.mean(sc))
            steered_means.append(steered_mean)
            lifts.append(steered_mean - baseline[key])
        per_layer[L] = {"mean_lift": float(np.mean(lifts)) if lifts else None,
                        "std_lift": float(np.std(lifts)) if lifts else None,
                        "mean_steered": float(np.mean(steered_means)) if steered_means else None,
                        "n_cells": len(lifts)}
        log.info(">>> layer %d: mean self-lift = %+.3f (n=%d)", L, per_layer[L]["mean_lift"], len(lifts))
    gen_f.close()
    judge.close()

    summary = {"model": Path(args.model).name, "criterion": "selfsteer_lift", "alpha": args.alpha,
               "judge_model": args.judge_model, "baseline_mean": float(np.mean(list(baseline.values()))),
               "rows": [{"layer": L, **per_layer[L]} for L in args.layers if L in per_layer]}
    json.dump(summary, open(out_dir / "selfsteer_lift.json", "w"), indent=2)
    print(f"\nwrote {out_dir / 'selfsteer_lift.json'}")
    best = max(summary["rows"], key=lambda r: r["mean_lift"])
    print(f">>> largest self-steer lift: layer {best['layer']}  ({best['mean_lift']:+.3f})")


if __name__ == "__main__":
    main()
