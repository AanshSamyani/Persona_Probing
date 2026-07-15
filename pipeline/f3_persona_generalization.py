#!/usr/bin/env python3
"""f3: in-context persona GENERALIZATION test on the base model, with an n-sweep.

Follow-up to the few-shot pilot (f2). There, demos and target were the SAME
persona, so "copy the demos" and "condition on the persona" were
indistinguishable. Here the demos are n DIFFERENT (seen) personas answering one
fixed question, and the target is a HELD-OUT persona described only by its
system prompt. Copying can't produce the held-out persona from OTHER personas'
demos — the base must read the held-out description. This separates task-format
(from the diverse demos) from persona-content (from the held-out description).

**n-sweep:** we vary the NUMBER of seen-persona demos n ∈ {1,3,5,7} (nested: n
uses the first n seen personas). This measures how many diverse examples the base
needs before it reliably keys on the held-out *description* rather than the
nearest demo. n=0 (description only, no demos) is kept as a baseline.

Prompt (raw text, base model; one fixed question, one trait per prompt):
    Persona: {seen_1 description}
    Q: {fixed question}
    A: {seen_1 on-policy answer}
    ... (n seen personas)
    Persona: {HELD-OUT description}
    Q: {fixed question}
    A:            <- base generates

Controls (inline):
  - n-sweep (1,3,5,7) — sample complexity of the description->voice mapping.
  - swap-description — held-out personas share the SAME demo block per (trait, q, n).
  - null / nonsense held-out "personas" — should read generic.
  - n=0 — description only, no demos (the pilot's failure mode).
  - shuffle — reversed demo order at max n (recency control).

Behavioral/feasibility only — no activation extraction. Reuses f1's instruct
demos. Outputs to results/persona_generalization/.

Usage:
    python pipeline/f3_persona_generalization.py --dry-run
    python pipeline/f3_persona_generalization.py \
        --demos-dir outputs/OLMo-2-1124-7B-Instruct/persona_gen/demos --n-values 1 3 5 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering.config import OLMO_2_7B, OUTPUTS_DIR, Trait
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_persona
from persona_steering.utils import log

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_BASE = Path("/workspace/models/OLMo-2-1124-7B")

SEEN = ["farmer", "drill_sergeant", "con_artist", "therapist",
        "tech_ceo", "kindergarten_teacher", "professor", "politician"]
HELDOUT_REAL = ["surgeon", "zen_master", "six_year_old"]     # distinctive voices
HELDOUT_CONTROL = ["null", "nonsense"]                        # should read as generic


def default_base_model() -> str:
    return str(LOCAL_BASE) if LOCAL_BASE.exists() else OLMO_2_7B.hf_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="In-context persona generalization (n-sweep) on the base model")
    p.add_argument("--base-model", default=default_base_model())
    p.add_argument("--demos-dir", default=None,
                   help="f1 output dir (default: outputs/OLMo-2-1124-7B-Instruct/persona_gen/demos)")
    p.add_argument("--seen", nargs="+", default=SEEN)
    p.add_argument("--heldout", nargs="+", default=HELDOUT_REAL)
    p.add_argument("--controls", nargs="+", default=HELDOUT_CONTROL)
    p.add_argument("--traits", nargs="+", default=None, help="default: traits found in demos _meta.json")
    p.add_argument("--n-values", nargs="+", type=int, default=[1, 3, 5, 7],
                   help="number of seen-persona demos to sweep (nested: first n seen personas)")
    p.add_argument("--n-fixed-questions", type=int, default=3)
    p.add_argument("--no-recency", dest="recency", action="store_false", default=True,
                   help="disable the reversed-order recency check at max n")
    p.add_argument("--demo-max-words", type=int, default=70)
    p.add_argument("--desc-max-words", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def flat(text: str) -> str:
    return " ".join(text.split())


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]).rstrip() + " …"


def load_response(demos_dir: Path, persona: str, trait: str, qid: int) -> str | None:
    """Instruct on-policy (trait-positive) response for persona to question qid."""
    path = demos_dir / f"{persona}_{trait}_pos.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["question_index"] == qid:
                return e["response"]
    return None


def persona_desc(slug: str, desc_max_words: int) -> str:
    try:
        d = load_persona(slug).default_system_prompt or ""
    except Exception:
        d = ""
    return truncate_words(flat(d), desc_max_words)


def format_block(demos, seen_descs, question, demo_max_words) -> str:
    """demos = ordered list of (slug, response)."""
    parts = []
    for slug, resp in demos:
        parts.append(f"Persona: {seen_descs[slug]}\nQ: {question}\nA: "
                     f"{truncate_words(flat(resp), demo_max_words)}")
    return "\n\n".join(parts)


def build_target(desc: str, question: str) -> str:
    return f"Persona: {desc}\nQ: {question}\nA:"


def clip_completion(text: str) -> str:
    for stop in ("\nPersona:", "\nQ:", "\nA:"):
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text.strip()


def batch_generate(model, tok, prompts, args, device):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    gk = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
              repetition_penalty=args.repetition_penalty,
              no_repeat_ngram_size=args.no_repeat_ngram_size)
    if args.temperature and args.temperature > 0:
        gk.update(do_sample=True, temperature=args.temperature, top_p=0.9)
    else:
        gk.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**enc, **gk)
    new = out[:, enc["input_ids"].shape[1]:]
    return [clip_completion(tok.decode(x, skip_special_tokens=True)) for x in new]


def main() -> None:
    args = parse_args()
    demos_dir = Path(args.demos_dir) if args.demos_dir else \
        OUTPUTS_DIR / "OLMo-2-1124-7B-Instruct" / "persona_gen" / "demos"
    meta_path = demos_dir / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        log.warning("No demos _meta.json at %s — run f1 first.", meta_path)
        meta = {"traits": ["assertiveness", "honesty"], "n_demo_questions": 8}

    traits = [Trait(t) for t in (args.traits or meta["traits"])]
    n_fixed = min(args.n_fixed_questions, meta.get("n_demo_questions", args.n_fixed_questions))
    sweep = sorted({n for n in args.n_values if n > 0})     # positive n only; n=0 handled as baseline
    heldout_all = list(args.heldout) + list(args.controls)
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / "persona_generalization"

    seen_descs = {s: persona_desc(s, args.desc_max_words) for s in args.seen}
    heldout_descs = {h: persona_desc(h, args.desc_max_words) for h in heldout_all}

    # ---- build all jobs -------------------------------------------------
    jobs = []  # dict(trait, qid, question, heldout, condition, n, prompt)

    def job(trait, qid, question, h, cond, n, prompt):
        return dict(trait=trait, qid=qid, question=question, heldout=h,
                    condition=cond, n=n, prompt=prompt)

    for trait in traits:
        ds = load_trait_dataset(trait)
        for qid in range(n_fixed):
            question = ds.questions[qid]
            avail = []
            for s in args.seen:
                r = load_response(demos_dir, s, trait.value, qid)
                if r is not None:
                    avail.append((s, r))
            if len(avail) < 2:
                log.warning("Only %d seen demos for %s q%d — skipping", len(avail), trait.value, qid)
                continue
            maxn = min(max(sweep) if sweep else 0, len(avail))
            for h in heldout_all:
                tgt = build_target(heldout_descs[h], question)
                jobs.append(job(trait.value, qid, question, h, "n0", 0, tgt))          # baseline
                for n in sweep:
                    nn = min(n, len(avail))
                    block = format_block(avail[:nn], seen_descs, question, args.demo_max_words)
                    jobs.append(job(trait.value, qid, question, h, f"n{n}", n, f"{block}\n\n{tgt}"))
                if args.recency and maxn >= 2:
                    block_rev = format_block(list(reversed(avail[:maxn])), seen_descs, question, args.demo_max_words)
                    jobs.append(job(trait.value, qid, question, h, f"n{maxn}_shuffle", maxn, f"{block_rev}\n\n{tgt}"))

    log.info("Persona-generalization plan: %d base generations | traits=%s fixed_q=%d seen=%d sweep=%s heldout=%s",
             len(jobs), [t.value for t in traits], n_fixed, len(args.seen), sweep, heldout_all)

    if args.dry_run:
        print(f"=== example prompt (largest n, first job of that kind) ===\n")
        ex = next((j for j in jobs if j["condition"] == f"n{max(sweep)}"), jobs[0] if jobs else None)
        print(ex["prompt"][:2200] if ex else "(no jobs — run f1 first)")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    log.info("Loading base model %s ...", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device).eval()

    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        gens = batch_generate(model, tok, [j["prompt"] for j in batch], args, device)
        for j, g in zip(batch, gens):
            j["generation"] = g
        log.info("  generated %d/%d", min(i + args.batch_size, len(jobs)), len(jobs))

    # ---- save -----------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "base_generations.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps({k: j[k] for k in
                                ("trait", "qid", "question", "heldout", "n", "condition", "generation")}) + "\n")
    write_markdown(jobs, traits, n_fixed, sweep, args, demos_dir, seen_descs, heldout_all, results_dir)
    log.info("Done. Eyeball: %s", results_dir / "generations.md")


def write_markdown(jobs, traits, n_fixed, sweep, args, demos_dir, seen_descs, heldout_all, results_dir):
    gmap = {}
    qtext = {}
    for j in jobs:
        gmap[(j["trait"], j["qid"], j["heldout"], j["condition"])] = j["generation"]
        qtext[(j["trait"], j["qid"])] = j["question"]
    maxn = min(max(sweep) if sweep else 0, len(args.seen))

    L = [
        "# In-context persona generalization (n-sweep) — base model", "",
        f"Base: `{args.base_model}`. Demos: n seen personas (from instruct, f1) answering one fixed "
        "question; target = a **held-out** persona described only by its system prompt.", "",
        "**The test:** copying can't produce a held-out persona from *other* personas' demos — the base "
        "must read the held-out description. **n-sweep** varies how many seen-persona demos it gets "
        f"(nested: n uses the first n of `{', '.join(args.seen)}`).", "",
        "- `n=0` = description only, no demos (pilot failure mode) · `nX` = X demos · "
        f"`n{maxn}_shuffle` = reversed order (recency control).",
        "- For each (trait, question) the held-out personas share the **same** demo block per n "
        "(swap-description control). `null`/`nonsense` should read **generic**.", "",
    ]
    row_conds = ["n0"] + [f"n{n}" for n in sweep]
    if args.recency and maxn >= 2:
        row_conds.append(f"n{maxn}_shuffle")

    for trait in [t.value for t in traits]:
        for qid in range(n_fixed):
            if (trait, qid) not in qtext:
                continue
            L.append(f"\n## {trait} · Q{qid}: {flat(qtext[(trait, qid)])}\n")
            L.append(f"<details><summary>seen demo pool (first {maxn}, in order; n=k uses the first k)</summary>\n")
            for i, s in enumerate(args.seen[:maxn], 1):
                r = load_response(demos_dir, s, trait, qid)
                if r is not None:
                    L.append(f"{i}. **{s}:** {flat(truncate_words(r, args.demo_max_words))}")
            L.append("\n</details>\n")
            for h in heldout_all:
                tag = " *(control)*" if h in args.controls else ""
                L.append(f"### held-out: {h}{tag}\n")
                ref = load_response(demos_dir, h, trait, qid)
                if ref is not None:
                    L.append(f"- **instruct ref:** {flat(truncate_words(ref, args.demo_max_words))}")
                for cond in row_conds:
                    g = gmap.get((trait, qid, h, cond))
                    if g is not None:
                        label = cond.replace("_shuffle", " (shuffle)").replace("n", "n=")
                        L.append(f"- **{label}:** {flat(g) or '—'}")
                L.append("")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "generations.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
