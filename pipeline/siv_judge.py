#!/usr/bin/env python3
"""siv_judge: LLM-judge the IV sanity generations (the paper's method).

Scores the base few-shot generations from siv_sanity with the paper's judge
(Claude Sonnet 4.5 via OpenRouter — the default in the repo's judge pipelines)
on the TRAIT axis (0..1), and optionally persona-match. This is the authoritative
trait-elicitation check (the SBERT AUROC was a judge-free proxy).

Key output — per (persona, trait): mean trait score for POS vs NEG generations.
If pos > neg the contrast is real; if pos ≈ neg it's noise (expected for honesty,
whose neg pole is refused → both directions read as honest).

Cost-aware:
  - `--max-per-cell` caps how many generations are judged per cell/direction.
  - trait-only by default; `--persona` adds persona-match (doubles the calls).
  - `--dry-run` just prints the call count + rough cost.
  - `--model anthropic/claude-haiku-4.5` for a ~3x cheaper (non-paper) judge.

Requires OPENROUTER_API_KEY.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python pipeline/siv_judge.py --dry-run
    python pipeline/siv_judge.py --max-per-cell 8
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from persona_steering.config import Trait
from persona_steering.openrouter_judge import OpenRouterJudge
from persona_steering.personas import load_persona
from persona_steering.utils import log

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "iv_sanity"
# rough per-call cost (input+output) for a short rating, USD
COST_PER_CALL = {"anthropic/claude-sonnet-4.5": 0.0015, "anthropic/claude-haiku-4.5": 0.0004}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-judge the IV sanity generations (trait pos-vs-neg)")
    p.add_argument("--generations", default=None, help="default: results/iv_sanity/generations.jsonl")
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5", help="paper's judge; OpenRouter slug")
    p.add_argument("--max-per-cell", type=int, default=8, help="gens judged per (persona,trait,direction)")
    p.add_argument("--persona", action="store_true", help="also judge persona-match (doubles calls)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gen_file = Path(args.generations) if args.generations else RESULTS / "generations.jsonl"
    records = [json.loads(l) for l in open(gen_file)]

    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for r in records:
        groups[(r["persona"], r["trait"], r["direction"])].append(r)
    sampled = []
    for recs in groups.values():
        rng.shuffle(recs)
        sampled.extend(recs[:args.max_per_cell])

    n_calls = len(sampled) * (2 if args.persona else 1)
    est = COST_PER_CALL.get(args.model, 0.0015) * n_calls
    print(f"Judge model: {args.model}")
    print(f"Cells: {len(groups)}  |  generations judged: {len(sampled)}  |  "
          f"total calls: {n_calls}  |  rough cost: ~${est:.2f}")
    if args.dry_run:
        return

    judge = OpenRouterJudge(model=args.model)
    pdesc = {}

    def desc(slug):
        if slug not in pdesc:
            try:
                pdesc[slug] = load_persona(slug).description or slug
            except Exception:
                pdesc[slug] = slug
        return pdesc[slug]

    def judge_one(r):
        t = Trait(r["trait"])
        text = r["generation"] or "(empty)"
        out = {k: r[k] for k in ("persona", "trait", "direction", "question", "generation")}
        out["trait_score"] = judge.score_trait(text, t).score
        out["persona_score"] = (judge.score_persona_match(text, r["persona"], desc(r["persona"])).score
                                if args.persona else None)
        return out

    log.info("judging %d generations with %s (%d workers)...", len(sampled), args.model, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        scored = list(ex.map(judge_one, sampled))
    judge.close()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "judge_scores.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    # --- aggregate: per (persona, trait) pos vs neg ---
    def cell(persona, trait, direction):
        return [s["trait_score"] for s in scored
                if s["persona"] == persona and s["trait"] == trait and s["direction"] == direction]

    personas = sorted({s["persona"] for s in scored})
    traits = sorted({s["trait"] for s in scored})

    L = ["# IV sanity — LLM-judge trait elicitation (paper's method)", "",
         f"Judge: `{args.model}`.  trait_score: 0 = strongly negative-pole, 1 = strongly positive-pole.", "",
         "Key column: **gap = mean(pos) − mean(neg)**. Large positive ⇒ the trait is genuinely "
         "elicited and the IV contrast is real; ≈0 ⇒ pos ≈ neg (contrast is noise, e.g. honesty).", "",
         "| Persona | Trait | pos | neg | gap | n(pos/neg) |", "|---|---|---|---|---|---|"]
    for persona in personas:
        for trait in traits:
            pos, neg = cell(persona, trait, "pos"), cell(persona, trait, "neg")
            if not pos or not neg:
                continue
            L.append(f"| {persona} | {trait} | {np.mean(pos):.2f} | {np.mean(neg):.2f} | "
                     f"{np.mean(pos)-np.mean(neg):+.2f} | {len(pos)}/{len(neg)} |")
    L += ["", "### mean gap per trait (averaged over personas)", "", "| Trait | mean gap | mean pos | mean neg |", "|---|---|---|---|"]
    for trait in traits:
        gaps, ps, ns = [], [], []
        for persona in personas:
            pos, neg = cell(persona, trait, "pos"), cell(persona, trait, "neg")
            if pos and neg:
                gaps.append(np.mean(pos) - np.mean(neg)); ps.append(np.mean(pos)); ns.append(np.mean(neg))
        if gaps:
            L.append(f"| {trait} | {np.mean(gaps):+.2f} | {np.mean(ps):.2f} | {np.mean(ns):.2f} |")

    if args.persona:
        L += ["", "## Persona-match (judge, 0..1)", "", "| Persona | mean persona-match | n |", "|---|---|---|"]
        for persona in personas:
            ps = [s["persona_score"] for s in scored if s["persona"] == persona and s["persona_score"] is not None]
            if ps:
                L.append(f"| {persona} | {np.mean(ps):.2f} | {len(ps)} |")

    (RESULTS / "judge_elicitation.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", RESULTS / "judge_elicitation.md")


if __name__ == "__main__":
    main()
