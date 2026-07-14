#!/usr/bin/env python3
"""Summarise the OLMo trajectory run into results/olmo_trajectory/SUMMARY.md.

Reads the JSON written by ``t3_trajectory_analysis.py`` and emits the two
headline tables from the paper:
  - mean shared-variance ratio per training stage  (Figure 19)
  - transfer-matrix Frobenius / Spearman between stages  (Table 6)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from persona_steering.config import OLMO_2_7B, OUTPUTS_DIR
from persona_steering.utils import model_short_name

STAGES = ["pretrain_1pct", "pretrain_10pct", "pretrain_50pct", "base", "sft", "dpo", "instruct"]
LABELS = {
    "pretrain_1pct": "Pretrain 1%", "pretrain_10pct": "Pretrain 10%",
    "pretrain_50pct": "Pretrain 50%", "base": "Base",
    "sft": "SFT", "dpo": "DPO", "instruct": "Instruct",
}
# Same comparisons the paper reports in Table 6.
COMPARISONS = [
    ("pretrain_1pct", "pretrain_10pct"), ("pretrain_10pct", "pretrain_50pct"),
    ("pretrain_50pct", "base"), ("base", "sft"), ("base", "instruct"),
    ("sft", "dpo"), ("dpo", "instruct"),
]


def main() -> None:
    tdir = OUTPUTS_DIR / model_short_name(OLMO_2_7B.hf_id) / "trajectory"
    var = json.load(open(tdir / "variance_trajectory.json"))
    dist = json.load(open(tdir / "transfer_matrix_distances.json"))
    meta = json.load(open(tdir / "trajectory_meta.json"))

    lines: list[str] = []
    lines.append("# OLMo-2 7B training-trajectory replication")
    lines.append("")
    lines.append("Paper Appendix F — Figure 19 (shared variance ratio) and Table 6 "
                 "(transfer-matrix distances).")
    lines.append("")
    lines.append(f"- Analysis layer: **{meta.get('layer')}**")
    lines.append(f"- Personas: {len(meta.get('personas', []))} — {', '.join(meta.get('personas', []))}")
    lines.append(f"- Traits: {len(meta.get('traits', []))} — {', '.join(meta.get('traits', []))}")
    lines.append("")

    lines.append("## Shared variance ratio per stage (Figure 19)")
    lines.append("")
    lines.append("| Stage | Mean shared variance |")
    lines.append("|---|---|")
    for s in [s for s in STAGES if s in var]:
        vals = list(var[s].values())
        m = np.mean(vals) if vals else float("nan")
        lines.append(f"| {LABELS[s]} | {m * 100:.1f}% |")
    lines.append("")
    lines.append("_Paper targets: 95.7 / 96.5 / 96.0 / 96.8 (pretrain→base) → "
                 "85.5 / 84.2 / 83.5 (SFT/DPO/Instruct)._")
    lines.append("")

    lines.append("## Transfer-matrix distances between stages (Table 6)")
    lines.append("")
    lines.append("| Comparison | Frobenius d_F | Spearman rho_s |")
    lines.append("|---|---|---|")
    for a, b in COMPARISONS:
        if a in dist and b in dist.get(a, {}):
            d = dist[a][b]
            lines.append(f"| {LABELS[a]} → {LABELS[b]} | "
                         f"{d.get('frobenius', float('nan')):.2f} | "
                         f"{d.get('spearman_rho', float('nan')):.2f} |")
    lines.append("")

    out_dir = Path(__file__).resolve().parent.parent / "results" / "olmo_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
