"""Splice an experiment into a base submission, so a leaderboard delta measures only the questions it re-ran.

    python runs/analysis/splice.py uncertain runs/sql_public > runs/analysis/uncertain_sql_public.txt
        ids the base run was unsure of: abstained, or needed the extra thinking samples (vote share < 0.75 on the first 8)
    python runs/analysis/splice.py merge runs/sql_public/submission.jsonl runs/sql_agent_unc runs/sql_public_agent_unc
        base rows, with every question the experiment run answered in its trace replaced by the run's row
Rows failing the format check become abstain. Prints how many rows changed and how.
"""
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "train"))
from check_submission import problem  # noqa: E402


def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


if sys.argv[1] == "uncertain":
    for t in read(R / sys.argv[2] / "trace.jsonl"):
        if t["final"]["type"] == "abstain" or len(t["samples"]) > 8:
            print(t["id"])
elif sys.argv[1] == "merge":
    base, run, out = R / sys.argv[2], R / sys.argv[3], R / sys.argv[4]
    rows = read(base)
    new = {t["id"]: {"id": t["id"], **t["final"]} for t in read(run / "trace.jsonl")}
    bad = [i for i, r in new.items() if problem(r)]
    for i in bad:
        new[i] = {"id": i, "type": "abstain"}
    merged = [new.get(r["id"], r) for r in rows]
    changed = [(a, b) for a, b in zip(rows, merged) if a != b]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "submission.jsonl").open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter("abstain -> answer" if a["type"] == "abstain" else "answer -> abstain" if b["type"] == "abstain" else "answer changed"
                    for a, b in changed)
    print(f"{len(new)} questions re-run, {len(changed)} rows changed {dict(kinds)}, {len(bad)} invalid rows abstained; "
          f"answered {sum(r['type'] != 'abstain' for r in rows)} -> {sum(r['type'] != 'abstain' for r in merged)} -> {out / 'submission.jsonl'}")
