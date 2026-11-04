"""Re-execute a run's logged programs on the current data layer, then postprocess, vote and decide as
Agent.answer_business does. No LLM: measures a data-layer or sandbox change on the exact same model outputs.

    .venv/bin/python train/replay_run.py runs/gemma_v5_public public_test runs/gemma_v6_replay_public

Writes OUT/submission.jsonl and OUT/logs/<id>.json (samples with the new execution results). On train, also
prints PA against answers.jsonl for the old and the new answers.
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
from datamate.sandbox import Executor  # noqa: E402
from datamate.scoring import score_one  # noqa: E402

g = {"__name__": "replay_run"}
run_lib_cells("02_agent.ipynb", g)


def settle(question: str, result) -> tuple[dict | None, str | None]:
    if not result.ok:
        return None, result.error
    try:
        return g["postprocess"](question, g["normalize_answer"](result.answer)), None
    except ValueError as e:
        return None, f"Invalid ANSWER: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("split")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    run, out = (REPO / args.run).resolve(), (REPO / args.out).resolve()
    cfg = json.loads((run / "config.json").read_text())
    k, tau = cfg["k"], cfg["tau"]
    root = g["split_root"](args.split)
    old = [json.loads(line) for line in (run / "submission.jsonl").open(encoding="utf-8")]
    logs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (run / "logs").glob("*.json")}
    answers_file = root / "answers.jsonl"
    refs = {r["id"]: r for r in g["read_jsonl"](answers_file)} if answers_file.exists() else None
    (out / "logs").mkdir(parents=True, exist_ok=True)
    by_biz: dict[str, list[dict]] = {}
    for r in old:
        by_biz.setdefault(logs[r["id"]]["business_id"], []).append(logs[r["id"]])
    new_rows, crashes = {}, Counter()
    for biz_id, lgs in sorted(by_biz.items()):
        biz = g["load_business_cached"](args.split, biz_id)          # builds the current cache version if missing
        jobs = [(s["code"], lg["as_of_date"]) for lg in lgs for s in lg["samples"] if s.get("code")]
        with Executor(g["sandbox_tables"](args.split, biz), root / "calendar.csv") as ex:
            results = iter(ex.run_many(jobs))
        for lg in lgs:
            first, repaired, by_index = [], [], {}
            for s in lg["samples"]:
                ans, err = settle(lg["question"], next(results)) if s.get("code") else (None, "No python code block")
                crashes["old"] += bool(s.get("error"))
                crashes["new"] += ans is None
                s.update(answer=ans, error=err)
                (repaired if s.get("repair") else first).append(ans)
                if "repair_of" in s:
                    by_index[s["repair_of"]] = ans
            answers = first + [None] * (k - len(first))
            if by_index:   # v7 logs: a repair takes the place of the sample it fixes
                for j, ans in by_index.items():
                    answers[j] = ans
            elif all(a is None for a in answers):
                fixed = next((a for a in repaired if a is not None), None)
                if fixed is not None:
                    answers = [fixed] + [None] * (k - 1)
            voted = g["vote"](answers, k)
            row = {"id": lg["id"], **g["decide"](voted, tau)}
            lg.update(final=row, support=voted["support"], clusters=voted["clusters"])
            new_rows[lg["id"]] = row
            log_path = out / "logs" / f"{lg['id']}.json"
            log_path.write_text(json.dumps(lg, ensure_ascii=False, indent=1), encoding="utf-8")
        answered = sum(new_rows[lg["id"]]["type"] != "abstain" for lg in lgs)
        print(f"{biz_id}: {len(lgs)} q, answered {answered}", flush=True)
    rows = [new_rows[r["id"]] for r in old]
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    with (out / "submission.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"failed samples: {crashes['old']} -> {crashes['new']} of {sum(len(lg['samples']) for lg in logs.values())}")
    print(f"answered: {sum(r['type'] != 'abstain' for r in old)} -> {sum(r['type'] != 'abstain' for r in rows)}"
          f" | changed answers {sum(a != b for a, b in zip(old, rows))}")
    if refs:
        for name, rr in [("old", old), ("new", rows)]:
            c = Counter(score_one(r, refs[r["id"]]) for r in rr)
            print(f"{name}: PA {(c[1] - c[-1]) / len(rr):+.3f} right {c[1]} wrong {c[-1]} abstain {c[0]}")
    print("DONE")


if __name__ == "__main__":
    main()
