#!/usr/bin/env python3
"""fig1_cos_to_null: reproduce Figure 1 from CAA vectors of ANY model.

Figure 1 = per-trait cosine of each persona-conditioned trait vector with the
default (null) trait vector:  cos(v_{trait,persona}, v_{trait,null}).  One violin
per trait (personas as dots, red diamond = persona mean), NONSENSE control as a
red ×, traits ordered by mean (most context-dependent on the left).

Parametrized version of scripts/build_paper_figures.py::fig1 (which hardcodes the
Gemma paths). Works on any model's CAA vectors.

Usage:
    python pipeline/fig1_cos_to_null.py \
      --vectors-dir outputs/Qwen2.5-7B-Instruct/caa_vectors \
      --layer 14 --model-name Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
            "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
TRAITS = ["assertiveness", "empathy", "risk_taking", "honesty",
          "confidence", "deference", "warmth", "impulsivity"]
TRAIT_LABEL = {t: t.replace("_", " ").title() for t in TRAITS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Figure 1 (cos to null) from CAA vectors")
    p.add_argument("--vectors-dir", required=True)
    p.add_argument("--layer", type=int, default=14, help="layer index into the (n_layers, hidden) vector")
    p.add_argument("--model-name", default="model")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_vec(vec_dir: Path, slug: str, trait: str, layer: int):
    path = Path(vec_dir) / f"{slug}_{trait}.pt"
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=True)
    v = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
    if v.ndim == 2:
        if layer >= v.shape[0]:
            return None
        return v[layer].float().numpy()
    return v.float().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    args = parse_args()
    vd, L = Path(args.vectors_dir), args.layer
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / f"fig1_{args.model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    table, nonsense = {}, {}
    for t in TRAITS:
        v_null = load_vec(vd, "null", t, L)
        if v_null is None:
            print(f"WARN: no null vector for {t} (need null_{t}.pt at layer {L}) — skipping trait")
            continue
        table[t] = {p: cos(vp, v_null) for p in PERSONAS if (vp := load_vec(vd, p, t, L)) is not None}
        vn = load_vec(vd, "nonsense", t, L)
        if vn is not None:
            nonsense[t] = cos(vn, v_null)

    present = [t for t in TRAITS if table.get(t)]
    if not present:
        raise SystemExit(f"No vectors found under {vd} at layer {L}. Check the path / run 3_vectors first.")
    means = {t: float(np.mean(list(table[t].values()))) for t in present}
    order = sorted(present, key=lambda t: means[t])          # most context-dependent left
    pos = np.arange(len(order))
    data = [list(table[t].values()) for t in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    parts = ax.violinplot(data, positions=pos, widths=0.75,
                          showmeans=False, showextrema=False, showmedians=False)
    for b in parts["bodies"]:
        b.set_facecolor("#cfd8dc"); b.set_edgecolor("#607d8b"); b.set_alpha(0.7)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jit = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos[i] + jit, vals, s=18, color="#37474f", edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([pos[i]], [np.mean(vals)], s=70, marker="D", color="#c0392b",
                   edgecolor="white", linewidth=1.0, zorder=4, label="persona mean" if i == 0 else None)
        if order[i] in nonsense:
            ax.scatter([pos[i]], [nonsense[order[i]]], s=80, marker="x", color="#e74c3c",
                       linewidth=2, zorder=5, label="nonsense" if i == 0 else None)
    ax.axhline(1.0, color="#1f7a1f", lw=0.8, ls=":")
    ax.set_xticks(pos)
    ax.set_xticklabels([TRAIT_LABEL[t] for t in order], rotation=20, ha="right")
    ax.set_ylabel(r"$\cos(\mathbf{v}_{T,c},\,\mathbf{v}_{T,\mathrm{null}})$")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(f"Figure 1 — context-conditioned trait vectors vs null\n"
                 f"{args.model_name}, layer {L}, CAA (10 personas × 8 traits)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_cos_to_null.png", dpi=180)
    plt.close(fig)
    (out_dir / "fig1_cos_to_null_table.json").write_text(
        json.dumps({"model": args.model_name, "layer": L, "cos_to_null": table, "nonsense": nonsense}, indent=2))

    print(f"Figure 1 -> {out_dir / 'fig1_cos_to_null.png'}")
    for t in order:
        print(f"  {t:14s} mean cos={means[t]:.3f}  nonsense={nonsense.get(t, float('nan')):.3f}")


if __name__ == "__main__":
    main()
