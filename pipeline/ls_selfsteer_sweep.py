#!/usr/bin/env python3
"""Layer selection, criterion (i): self-steering behavioral-lift sweep  (paper §A.2/§A.3).

Reproduces the paper's primary layer-selection criterion, extended over steering
strength: for each (candidate layer L, steering coefficient alpha), steer the
persona-prompted model with the cell's OWN trait vector (L2-normalized per §A.3),
generate, and score the trait with a held-out Claude judge. Lift = mean(steered
trait score) - mean(baseline trait score).

The paper picks the layer at alpha=2 whose lift is "among the largest in the sweep"
AND whose bootstrap stability (ls_bootstrap_sweep.py) is high; it also discusses
alpha in {1,2,4,8} (§A.3: alpha=2 chosen; 4 starts losing coherence; 8 collapses).
Sweeping alpha here reproduces that rationale on Qwen and shows whether the best
layer is alpha-dependent.

Every rollout (baseline + each layer x alpha) is streamed to generations.jsonl and
also written to a human-readable rollouts.md grouped by cell, so the actual text can
be eyeballed later (coherence, style, over-steering artifacts).

Design: loads the model ONCE, caches each cell's baseline (layer/alpha-independent),
computes normalized steer vectors per layer once (reused across alpha). Steering uses
the same assistant_axis.ActivationSteering path as pipeline/8_steered_generation.py,
so the alpha/normalize semantics match the paper. Judged via OpenRouterJudge.

Cost: judge calls = n_cells * n_questions * (1 + n_layers * n_alphas). Use --dry-run.

Requires OPENROUTER_API_KEY. Runs on the GPU server (needs the model + assistant-axis-ref).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python pipeline/ls_selfsteer_sweep.py \
      --model /workspace/models/Qwen2.5-7B-Instruct \
      --vectors-dir outputs/Qwen2.5-7B-Instruct/caa_vectors \
      --layers 6 10 14 16 18 19 20 21 23 26 \
      --alphas 1 2 4 8 --n-questions 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))                       # so `persona_steering` imports without pip install -e .
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
    p = argparse.ArgumentParser(description="Self-steer behavioral-lift layer x alpha sweep (criterion i)")
    p.add_argument("--model", required=True, help="HF id or local path (Qwen2.5-7B-Instruct)")
    p.add_argument("--vectors-dir", required=True, help="CAA vectors dir ({persona}_{trait}.pt)")
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0],
                   help="steering coefficients (paper discusses 1,2,4,8; selection uses 2)")
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


def write_rollouts_md(path: Path, records: list, baseline_mean: dict) -> None:
    """Human-readable dump grouped by cell -> question -> (baseline, then layer/alpha rollouts)."""
    by_cell = defaultdict(list)
    for r in records:
        by_cell[(r["persona"], r["trait"])].append(r)
    lines = ["# Self-steer layer×alpha rollouts (eyeball)", "",
             "Grouped by cell. For each question: baseline first, then steered rollouts "
             "sorted by (layer, alpha). `score` = judge trait score 0..1.", ""]
    for (persona, trait), recs in sorted(by_cell.items()):
        bm = baseline_mean.get((persona, trait))
        lines.append(f"\n## {persona} · {trait}" + (f"  (baseline mean {bm:.2f})" if bm is not None else ""))
        by_q = defaultdict(list)
        for r in recs:
            by_q[r["question"]].append(r)
        for q, qrecs in by_q.items():
            lines.append(f"\n**Q: {q}**\n")
            base = [r for r in qrecs if r["phase"] == "baseline"]
            for r in base:
                lines.append(f"- _baseline_ (score {r['trait_score']:.2f}): {r['response']}")
            steer = sorted([r for r in qrecs if r["phase"] == "steered"],
                           key=lambda r: (r["layer"], r["alpha"]))
            for r in steer:
                lines.append(f"- L{r['layer']} α={r['alpha']:g} (score {r['trait_score']:.2f}): {r['response']}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    n_steer_cfg = len(args.layers) * len(args.alphas)
    judge_calls = len(cells) * n_q * (1 + n_steer_cfg)
    est = COST_PER_CALL.get(args.judge_model, 0.0015) * judge_calls

    print(f"Layers:   {args.layers}")
    print(f"Alphas:   {args.alphas}")
    print(f"Cells:    {len(cells)} ({len(personas)} personas x {len(traits)} traits, vectors present)")
    print(f"Questions/cell: {n_q}   normalize={not args.no_normalize}")
    print(f"Generations: baseline {len(cells)*n_q} + steered {len(cells)*n_steer_cfg*n_q}")
    print(f"Judge model: {args.judge_model}")
    print(f"Judge calls: {judge_calls}   rough cost: ~${est:.2f}")
    if args.dry_run:
        return

    from assistant_axis.internals.model import ProbingModel
    from assistant_axis.steering import ActivationSteering
    from assistant_axis.generation import generate_response, format_conversation

    log.info("loading model %s ...", args.model)
    probing = ProbingModel(args.model)
    model, tok = probing.model, probing.tokenizer
    judge = OpenRouterJudge(model=args.judge_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_f = open(out_dir / "generations.jsonl", "a")
    records = []  # kept in memory for the readable md

    def emit(rec):
        gen_f.write(json.dumps(rec) + "\n")
        gen_f.flush()
        records.append(rec)

    def gen(system_prompt, question, steer=None, layer=None, alpha=None):
        conv = format_conversation(system_prompt, question, tok)
        if steer is None:
            return generate_response(model, tok, conv, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature)
        with ActivationSteering(model, steering_vectors=[steer], coefficients=[alpha],
                                layer_indices=[layer]):
            return generate_response(model, tok, conv, max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature)

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
            emit({"phase": "baseline", "layer": None, "alpha": 0.0, "persona": p, "trait": t.value,
                  "question": q, "response": r, "trait_score": s})
        baseline[(p, t.value)] = float(np.mean(sc))
        log.info("baseline %s/%s = %.3f", p, t.value, baseline[(p, t.value)])

    # ---- steered: layer x alpha ----
    per_cfg = {}
    for L in args.layers:
        sv = steer_vecs_at(L)
        for a in args.alphas:
            lifts, steered_means = [], []
            for p, t in cells:
                key = (p, t.value)
                if key not in sv:
                    continue
                sp = pmap[p].default_system_prompt
                sc = []
                for q in qs_by_trait[t.value]:
                    r = gen(sp, q, steer=sv[key].to(model.device), layer=L, alpha=a)
                    s = judge.score_trait(r, t).score
                    sc.append(s)
                    emit({"phase": "steered", "layer": L, "alpha": a, "persona": p, "trait": t.value,
                          "question": q, "response": r, "trait_score": s})
                sm = float(np.mean(sc))
                steered_means.append(sm)
                lifts.append(sm - baseline[key])
            per_cfg[(L, a)] = {"layer": L, "alpha": a,
                               "mean_lift": float(np.mean(lifts)) if lifts else None,
                               "std_lift": float(np.std(lifts)) if lifts else None,
                               "mean_steered": float(np.mean(steered_means)) if steered_means else None,
                               "n_cells": len(lifts)}
            log.info(">>> L%d α=%g: mean self-lift = %+.3f (n=%d)",
                     L, a, per_cfg[(L, a)]["mean_lift"], len(lifts))
    gen_f.close()
    judge.close()

    write_rollouts_md(out_dir / "rollouts.md", records, baseline)

    summary = {"model": Path(args.model).name, "criterion": "selfsteer_lift",
               "alphas": args.alphas, "layers": args.layers, "judge_model": args.judge_model,
               "baseline_mean": float(np.mean(list(baseline.values()))),
               "rows": [per_cfg[(L, a)] for L in args.layers for a in args.alphas if (L, a) in per_cfg]}
    json.dump(summary, open(out_dir / "selfsteer_lift.json", "w"), indent=2)
    print(f"\nwrote {out_dir / 'selfsteer_lift.json'}  and  {out_dir / 'rollouts.md'}")

    # report best per alpha
    for a in args.alphas:
        rows_a = [r for r in summary["rows"] if r["alpha"] == a and r["mean_lift"] is not None]
        if rows_a:
            best = max(rows_a, key=lambda r: r["mean_lift"])
            print(f"  α={a:g}: largest lift at layer {best['layer']}  ({best['mean_lift']:+.3f})")


if __name__ == "__main__":
    main()
