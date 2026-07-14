#!/usr/bin/env python3
"""f2: few-shot the BASE model with instruct demos and dump generations to eyeball.

Design-B feasibility pilot. Question: does the base model's persona+trait
generation get better as we add n in-context demonstrations (from the instruct
model, via f1)? Persona is conveyed via an explicit `Context:` line PLUS the n
demos (per the pilot spec). Controls: n=0 baseline and mismatched-persona demos.

Prompt shown to the base model (raw text, no chat template):

    Context: {persona system prompt}

    Question: {demo q1}
    Answer: {instruct demo response 1}
    ... (n demos)
    Question: {held-out question}
    Answer:            <- base model continues here

This is FEASIBILITY ONLY: we look at the generations, we do not extract
activations or build vectors yet. Outputs go to results/fewshot_pilot/.

Usage:
    python pipeline/f2_fewshot_base_generate.py --dry-run
    python pipeline/f2_fewshot_base_generate.py --directions pos
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering.config import OLMO_2_7B, OUTPUTS_DIR, Trait
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_persona
from persona_steering.utils import log, model_short_name

BASE_MODEL = OLMO_2_7B.hf_id  # allenai/OLMo-2-1124-7B
INSTRUCT_SHORT = "OLMo-2-1124-7B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent.parent
# Each pilot persona's "mismatch" demo source — a deliberately different voice.
MISMATCH = {"drill_sergeant": "farmer", "farmer": "drill_sergeant", "con_artist": "farmer"}
# Fallback defaults if the f1 _meta.json isn't present (e.g. a standalone --dry-run).
DEFAULT_META = {"personas": ["drill_sergeant", "con_artist", "farmer"],
                "traits": ["assertiveness", "honesty"],
                "n_demo_questions": 8, "n_target_questions": 4}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot the base model with instruct demos")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--demos-dir", default=None, help="f1 output dir (default: outputs/<instruct>/fewshot_pilot/demos)")
    p.add_argument("--n-values", nargs="+", type=int, default=[0, 1, 3, 5, 7])
    p.add_argument("--directions", nargs="+", default=["pos"], choices=["pos", "neg"])
    p.add_argument("--demo-max-words", type=int, default=90, help="truncate each demo response (context budget)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (reproducible reads)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mismatch", dest="include_mismatch", action="store_false", default=True)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]).rstrip() + " …"


def flat(text: str) -> str:
    """Collapse whitespace so multi-line answers stay on one markdown line."""
    return " ".join(text.split())


def load_demos(demos_dir: Path, persona: str, trait: str, direction: str):
    """Return sorted list of (question_index, question, response)."""
    path = demos_dir / f"{persona}_{trait}_{direction}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            out.append((e["question_index"], e["question"], e["response"]))
    out.sort(key=lambda x: x[0])
    return out


def build_prompt(persona_sys: str, demos, target_q: str, demo_max_words: int) -> str:
    parts = [f"Context: {persona_sys}", ""]
    for q, r in demos:
        parts.append(f"Question: {q}")
        parts.append(f"Answer: {truncate_words(r, demo_max_words)}")
        parts.append("")
    parts.append(f"Question: {target_q}")
    parts.append("Answer:")
    return "\n".join(parts)


def clip_completion(text: str) -> str:
    for stop in ("\nQuestion:", "\nContext:", "\nAnswer:"):
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text.strip()


def batch_generate(model, tok, prompts, max_new_tokens, temperature, device):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    new = out[:, enc["input_ids"].shape[1]:]
    return [clip_completion(tok.decode(x, skip_special_tokens=True)) for x in new]


def build_jobs(demos_dir, personas, traits, n_demo, n_target, args):
    """One job per (persona, trait, direction, n, condition, target question)."""
    jobs = []
    for slug in personas:
        persona = load_persona(slug)
        psys = persona.system_prompt_variants[0].strip()
        for trait in traits:
            ds = load_trait_dataset(trait)
            targets = list(enumerate(ds.questions))[n_demo:n_demo + n_target]  # (qid, text)
            for direction in args.directions:
                matched = load_demos(demos_dir, slug, trait.value, direction)
                mm_slug = MISMATCH.get(slug)
                mismatch = load_demos(demos_dir, mm_slug, trait.value, direction) if mm_slug else []
                for n in args.n_values:
                    conds = [("matched", slug, matched)]
                    if args.include_mismatch and n > 0 and mismatch:
                        conds.append(("mismatch", mm_slug, mismatch))
                    for cond, dpersona, dpool in conds:
                        demos = [(q, r) for _, q, r in dpool[:n]]
                        for qid, tq in targets:
                            jobs.append({
                                "persona": slug, "trait": trait.value, "direction": direction,
                                "n": n, "condition": cond, "demo_persona": dpersona,
                                "target_qid": qid, "target_q": tq,
                                "prompt": build_prompt(psys, demos, tq, args.demo_max_words),
                            })
    return jobs


def write_markdown(jobs, personas, trait_strs, directions, n_values, results_dir, demos_dir, demo_max_words, base_model):
    gmap = {}
    targets_map = defaultdict(dict)
    for j in jobs:
        gmap[(j["persona"], j["trait"], j["direction"], j["n"], j["condition"], j["target_qid"])] = j.get("generation", "")
        targets_map[(j["persona"], j["trait"], j["direction"])][j["target_qid"]] = j["target_q"]

    L = [
        "# Few-shot base-model pilot — generations", "",
        f"Base model: `{base_model}`. Demos from the instruct model (f1). "
        "Persona conveyed via `Context:` + n demos.", "",
        f"n-sweep: {n_values}. Read each held-out question top-to-bottom: does the answer get more "
        "coherent, more **in-persona**, and more **trait-expressing** as n grows?", "",
        "`mismatch` keeps the same `Context:` persona but pulls demos from a *different* persona — "
        "if the answer follows the demo persona rather than the Context, that's in-context copying, "
        "not the base model conditioning on the persona.", "",
    ]
    for slug in personas:
        for trait in trait_strs:
            for direction in directions:
                key = (slug, trait, direction)
                if key not in targets_map:
                    continue
                L.append(f"\n## {slug} · {trait} · {direction}\n")
                demos = load_demos(demos_dir, slug, trait, direction)
                if demos:
                    L.append(f"<details><summary>demos used (matched · {slug}, truncated)</summary>\n")
                    for _, q, r in demos[:max(n_values)]:
                        L.append(f"- **Q:** {flat(q)}<br>**A:** {flat(truncate_words(r, demo_max_words))}")
                    L.append("\n</details>\n")
                for qid, tq in sorted(targets_map[key].items()):
                    L.append(f"### held-out Q: {flat(tq)}\n")
                    L.append(f"**matched** (Context = demos = `{slug}`):\n")
                    for n in n_values:
                        g = gmap.get((slug, trait, direction, n, "matched", qid), "—")
                        L.append(f"- **n={n}:** {flat(g) or '—'}")
                    if any((slug, trait, direction, n, "mismatch", qid) in gmap for n in n_values):
                        L.append(f"\n**mismatch** (Context = `{slug}`, demos = `{MISMATCH.get(slug)}`):\n")
                        for n in n_values:
                            if n == 0:
                                continue
                            g = gmap.get((slug, trait, direction, n, "mismatch", qid))
                            if g is not None:
                                L.append(f"- **n={n}:** {flat(g) or '—'}")
                    L.append("")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "generations.md").write_text("\n".join(L))


def main() -> None:
    args = parse_args()
    demos_dir = Path(args.demos_dir) if args.demos_dir else OUTPUTS_DIR / INSTRUCT_SHORT / "fewshot_pilot" / "demos"
    meta_path = demos_dir / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        log.warning("No f1 _meta.json at %s — using defaults (run f1 first for real demos).", meta_path)
        meta = DEFAULT_META
    personas = meta["personas"]
    traits = [Trait(t) for t in meta["traits"]]
    n_demo, n_target = meta["n_demo_questions"], meta["n_target_questions"]

    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / "fewshot_pilot"

    jobs = build_jobs(demos_dir, personas, traits, n_demo, n_target, args)
    log.info("Base few-shot plan: %d generations · personas=%s traits=%s directions=%s n=%s",
             len(jobs), personas, [t.value for t in traits], args.directions, args.n_values)

    if args.dry_run:
        print("=== example prompt (first job) ===\n")
        print(jobs[0]["prompt"][:1600] if jobs else "(no jobs — demos missing)")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    log.info("Loading base model %s ...", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device).eval()

    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        gens = batch_generate(model, tok, [j["prompt"] for j in batch],
                              args.max_new_tokens, args.temperature, device)
        for j, g in zip(batch, gens):
            j["generation"] = g
        log.info("  generated %d/%d", min(i + args.batch_size, len(jobs)), len(jobs))

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "base_generations.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps({k: j[k] for k in (
                "persona", "trait", "direction", "n", "condition", "demo_persona",
                "target_qid", "target_q", "generation")}) + "\n")
    write_markdown(jobs, personas, [t.value for t in traits], args.directions,
                   args.n_values, results_dir, demos_dir, args.demo_max_words, args.base_model)

    log.info("Done. Eyeball: %s", results_dir / "generations.md")
    log.info("Raw: %s", results_dir / "base_generations.jsonl")


if __name__ == "__main__":
    main()
