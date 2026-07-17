#!/usr/bin/env python3
"""Combine the two paper layer-selection criteria and pick the Qwen layer.

Reads:
  - results/layer_select/bootstrap_{model}.json     (criterion ii: stability)
  - results/layer_select/selfsteer/selfsteer_lift.json  (criterion i: self-lift)

The paper (§A.2) selects a layer whose self-steer lift is "among the largest in
the sweep" AND whose bootstrap stability is high. We rank each criterion across
the swept layers, min-max normalise to [0,1], and score each layer by the mean of
the two normalised values (both must be high — a layer that steers well but is
unstable, or is rock-stable but doesn't steer, loses).

Usage:
    python pipeline/ls_select_layer.py --model-name Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LS = REPO_ROOT / "results" / "layer_select"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine bootstrap-stability + self-lift, pick the layer")
    p.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    p.add_argument("--bootstrap", default=None)
    p.add_argument("--selfsteer", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def norm01(d):
    vals = np.array(list(d.values()), float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def main() -> None:
    args = parse_args()
    bpath = Path(args.bootstrap) if args.bootstrap else LS / f"bootstrap_{args.model_name}.json"
    spath = Path(args.selfsteer) if args.selfsteer else LS / "selfsteer" / "selfsteer_lift.json"
    out_dir = Path(args.output_dir) if args.output_dir else LS
    out_dir.mkdir(parents=True, exist_ok=True)

    stab = {r["layer"]: r["stability"] for r in json.load(open(bpath))["rows"]} if bpath.exists() else {}
    lift = {r["layer"]: r["mean_lift"] for r in json.load(open(spath))["rows"]} if spath.exists() else {}
    if not stab:
        print(f"WARN: no bootstrap json at {bpath}")
    if not lift:
        print(f"WARN: no selfsteer json at {spath}")

    layers = sorted(set(stab) | set(lift))
    ns = norm01({L: stab[L] for L in layers if L in stab})
    nl = norm01({L: lift[L] for L in layers if L in lift})

    print(f"{'layer':>5} {'stability':>10} {'self-lift':>10} {'score':>8}")
    combined = {}
    for L in layers:
        parts = [x for x in (ns.get(L), nl.get(L)) if x is not None]
        score = float(np.mean(parts)) if parts else float("nan")
        combined[L] = score
        print(f"{L:5d} {stab.get(L, float('nan')):10.4f} {lift.get(L, float('nan')):+10.3f} {score:8.3f}")

    valid = {L: s for L, s in combined.items() if not np.isnan(s)}
    pick = max(valid, key=valid.get) if valid else None
    if pick is not None:
        print(f"\n>>> selected layer: {pick}  (stability={stab.get(pick)}, self-lift={lift.get(pick):+.3f})")

    json.dump({"model": args.model_name, "stability": stab, "self_lift": lift,
               "combined_score": combined, "selected_layer": pick},
              open(out_dir / f"selected_{args.model_name}.json", "w"), indent=2)

    # plot
    if layers:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax2 = ax1.twinx()
        Ls = [L for L in layers if L in lift]
        Lb = [L for L in layers if L in stab]
        if Ls:
            ax1.plot(Ls, [lift[L] for L in Ls], "o-", color="#1f77b4", lw=2, label="self-steer lift")
        if Lb:
            ax2.plot(Lb, [stab[L] for L in Lb], "s--", color="#d62728", lw=2, label="bootstrap stability")
        if pick is not None:
            ax1.axvline(pick, color="#2ca02c", ls=":", lw=1.5)
            ax1.annotate(f"selected L{pick}", (pick, ax1.get_ylim()[1]), color="#2ca02c",
                         fontsize=9, fontweight="bold", ha="center", va="top")
        ax1.set_xlabel("layer"); ax1.set_ylabel("self-steer trait-lift over baseline", color="#1f77b4")
        ax2.set_ylabel("bootstrap stability (pairwise cos)", color="#d62728")
        ax1.set_title(f"Layer selection ({args.model_name}) — paper's two criteria (§A.2)")
        ax1.grid(alpha=0.25, ls=":")
        fig.tight_layout()
        fig.savefig(out_dir / f"layer_select_{args.model_name}.png", dpi=170)
        plt.close(fig)
        print(f"wrote {out_dir / f'layer_select_{args.model_name}.png'}")


if __name__ == "__main__":
    main()
