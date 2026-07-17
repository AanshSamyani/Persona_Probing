#!/usr/bin/env python3
"""Layer selection, criterion (ii): bootstrap stability sweep  (paper §A.2 / §B.1).

For every candidate layer, re-extract each (persona, trait) CAA vector from
bootstrap-resampled contrastive pairs (n=50, with replacement) and measure the
within-cell pairwise cosine — the "noise floor" (how much the vector moves if we
resampled the questions). The paper wants a layer where this stability is high
(≈0.99 at Gemma L22). Pure numpy on the stored per-question CAA activations — NO
model needed.

Single pass: resamples once per cell and evaluates ALL candidate layers, instead
of reloading activations per layer like r1_bootstrap_vectors.py.

Usage:
    python pipeline/ls_bootstrap_sweep.py \
      --activations-dir outputs/Qwen2.5-7B-Instruct/caa_activations \
      --layers 10 14 16 18 19 20 21 22 24 26 \
      --model-name Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from persona_steering.utils import discover_activation_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap-stability layer sweep (criterion ii)")
    p.add_argument("--activations-dir", required=True,
                   help="CAA per-question activations ({persona}_{trait}_{pos|neg}.pt)")
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--n-bootstraps", type=int, default=50)
    p.add_argument("--personas", nargs="+", default=PERSONAS,
                   help="restrict to these personas (default: 10 core)")
    p.add_argument("--model-name", default="model")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def boot_vectors(pos_acts, neg_acts, n_boot, rng):
    """Return list of n_boot contrastive vectors, each (n_layers, hidden)."""
    clean = lambda t: torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)
    pos = [clean(a) for a in pos_acts]
    neg = [clean(a) for a in neg_acts]
    out = []
    for _ in range(n_boot):
        pi = rng.choice(len(pos), size=len(pos), replace=True)
        ni = rng.choice(len(neg), size=len(neg), replace=True)
        v = torch.stack([pos[i] for i in pi]).mean(0) - torch.stack([neg[i] for i in ni]).mean(0)
        out.append(v)
    return torch.stack(out)  # (n_boot, n_layers, hidden)


def pairwise_cos_at_layer(boots_layer):
    """Mean pairwise cosine among rows of (n_boot, hidden)."""
    x = boots_layer / (boots_layer.norm(dim=-1, keepdim=True) + 1e-9)
    sims = x @ x.T
    n = x.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(sims[iu[0], iu[1]].mean())


def main() -> None:
    args = parse_args()
    ad = Path(args.activations_dir)
    rng = np.random.default_rng(args.seed)

    pairs = [(p, t, pp, np_) for (p, t, pp, np_) in discover_activation_pairs(ad)
             if p in set(args.personas)]
    if not pairs:
        raise SystemExit(f"No activation pairs under {ad}. (Need CAA activations on the server.)")
    print(f"{len(pairs)} cells, {args.n_bootstraps} bootstraps, layers {args.layers}")

    per_layer = defaultdict(list)  # layer -> [pairwise cosine per cell]
    for persona, trait, pos_path, neg_path in pairs:
        pos = list(torch.load(pos_path, map_location="cpu", weights_only=True).values())
        neg = list(torch.load(neg_path, map_location="cpu", weights_only=True).values())
        if len(pos) < 2 or len(neg) < 2:
            continue
        boots = boot_vectors(pos, neg, args.n_bootstraps, rng)  # (n_boot, n_layers, hidden)
        for L in args.layers:
            if L < boots.shape[1]:
                per_layer[L].append(pairwise_cos_at_layer(boots[:, L, :]))

    rows = []
    print(f"\n{'layer':>5} {'stability':>10} {'min_cell':>9}  (mean within-cell pairwise cosine)")
    for L in args.layers:
        vals = per_layer.get(L, [])
        if not vals:
            continue
        m, mn = float(np.mean(vals)), float(np.min(vals))
        rows.append({"layer": L, "stability": round(m, 4), "min_cell": round(mn, 4), "n_cells": len(vals)})
        print(f"{L:5d} {m:10.4f} {mn:9.4f}")

    out = Path(args.output) if args.output else REPO_ROOT / "results" / "layer_select" / f"bootstrap_{args.model_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model_name, "criterion": "bootstrap_stability",
               "n_bootstraps": args.n_bootstraps, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    if rows:
        best = max(rows, key=lambda r: r["stability"])
        print(f">>> highest stability: layer {best['layer']}  ({best['stability']:.4f})")


if __name__ == "__main__":
    main()
