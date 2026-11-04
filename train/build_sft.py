"""Rejection-sampling SFT data from a 02_agent run on train: keep the sampled replies whose executed answer
matches answers.jsonl, rebuild the exact prompt the agent sent, write one JSON line per kept reply.

    .venv/bin/python train/build_sft.py runs/gen_rest data/sft/rest.jsonl [--per-question 2] [--exclude splits/holdout_shops.txt]

Run from the repo root with the inference venv (it loads the notebooks' lib cells for prompts and scoring).
Lines: {"id", "business_id", "system", "user", "reply"}; the system prompt repeats per shop, so the file is
compressed well by train_lora.py's grouping, not here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO / "notebooks")   # run_lib_cells and the relative paths in the notebooks expect it

from datamate.nblib import run_lib_cells  # noqa: E402
from datamate.scoring import score_one  # noqa: E402

g = {"__name__": "build_sft"}
run_lib_cells("02_agent.ipynb", g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-question", type=int, default=2, help="distinct correct replies kept per question")
    ap.add_argument("--exclude", type=Path, help="file of business ids never to train on")
    ap.add_argument("--prompt", default="v3")
    args = ap.parse_args()
    run, out = (REPO / args.run).resolve(), (REPO / args.out).resolve()
    exclude = set((REPO / args.exclude).read_text().split()) if args.exclude else set()

    refs = {r["id"]: r for r in g["read_jsonl"](g["split_root"]("train") / "answers.jsonl")}
    rulebook = (g["split_root"]("train") / "rulebook.md").read_text(encoding="utf-8")
    spec, systems = g["PROMPTS"][args.prompt], {}
    stats, rows = Counter(), []
    for path in sorted((run / "logs").glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        biz_id = log["business_id"]
        if biz_id in exclude:
            stats["excluded"] += 1
            continue
        stats["questions"] += 1
        good = []
        for s in log["samples"]:
            if s.get("repair") or not s.get("answer"):
                continue
            ans = g["postprocess"](log["question"], s["answer"])
            if score_one(ans, refs[log["id"]]) == 1:
                good.append(s["reply"])
        if not good:
            continue
        stats["solved"] += 1
        if biz_id not in systems:
            systems[biz_id] = spec.build_system(rulebook, g["describe_business"](g["load_business_cached"]("train", biz_id)))
        for reply in sorted(set(good), key=len)[:args.per_question]:
            rows.append({"id": log["id"], "business_id": biz_id, "system": systems[biz_id],
                         "user": log["user_prompt"], "reply": reply})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{stats['solved']} / {stats['questions']} questions with a correct sample, {len(rows)} replies -> {out}"
          f" (excluded {stats['excluded']} from held-out shops)")


if __name__ == "__main__":
    main()
