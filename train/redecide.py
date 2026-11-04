"""Rebuild a run's submission from its per-question logs with the current postprocess / vote / decide, without
the LLM: every sampled answer is in the logs. Mirrors Agent.answer_business: postprocess each sample; if every
first-round sample failed, the repaired answer counts as one sample of k; vote; keep it if support >= tau.

    .venv/bin/python train/redecide.py runs/gemma_v5_public runs/gemma_v5_public_pp [--tau 0.6]

Writes OUT/submission.jsonl in the run's question order and prints what changed.
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
os.chdir(REPO / "notebooks")

from datamate.nblib import run_lib_cells  # noqa: E402

g = {"__name__": "redecide"}
run_lib_cells("02_agent.ipynb", g)


def redecide(log: dict, k: int, tau: float) -> dict:
    first = [s for s in log["samples"] if not s.get("repair")]
    answers = [g["postprocess"](log["question"], s["answer"]) if s.get("answer") else None for s in first]
    repairs = [s for s in log["samples"] if s.get("repair")]
    if any("repair_of" in s for s in repairs):   # v7: a repair takes the place of the sample it fixes
        for s in repairs:
            answers[s["repair_of"]] = g["postprocess"](log["question"], s["answer"]) if s.get("answer") else None
    elif all(a is None for a in answers):
        for s in log["samples"]:
            if s.get("repair") and s.get("answer"):
                fixed = g["postprocess"](log["question"], s["answer"])
                if fixed is not None:
                    answers = [fixed] + [None] * (k - 1)
    answers += [None] * (k - len(answers))
    return {"id": log["id"], **g["decide"](g["vote"](answers, k), tau)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tau", type=float, default=None, help="default: the run's config.json")
    args = ap.parse_args()
    run, out = (REPO / args.run).resolve(), (REPO / args.out).resolve()
    cfg = json.loads((run / "config.json").read_text())
    k, tau = cfg["k"], cfg["tau"] if args.tau is None else args.tau
    old = [json.loads(line) for line in (run / "submission.jsonl").open(encoding="utf-8")]
    logs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (run / "logs").glob("*.json")}
    new = [redecide(logs[r["id"]], k, tau) for r in old]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "submission.jsonl").open("w", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    changed = [(a, b) for a, b in zip(old, new) if a != b]
    print(f"k={k} tau={tau}: {len(changed)} of {len(new)} answers changed -> {out / 'submission.jsonl'}")
    print("types before:", dict(Counter(r["type"] for r in old)))
    print("types after: ", dict(Counter(r["type"] for r in new)))
    for a, b in changed:
        print(f"  {a['id']} {logs[a['id']]['question'][:70]!r}: {a.get('type')} {str(a.get('answer'))[:40]} -> "
              f"{b.get('type')} {str(b.get('answer'))[:40]}")


if __name__ == "__main__":
    main()
