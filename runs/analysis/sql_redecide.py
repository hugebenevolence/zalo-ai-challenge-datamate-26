"""Rebuild an SQL-baseline submission from its trace with another vote-share threshold, no LLM:
python runs/analysis/sql_redecide.py runs/sql_public runs/sql_public_t0375 --threshold 0.375 [--split public_test]"""
import argparse
import importlib.util
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("predict", R / "sqlbase/predict.py")
import sys
sys.path[:0] = [str(R / "sqlbase"), str(R / "train")]
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from check_submission import problem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--threshold", type=float, required=True)
ap.add_argument("--split", default="public_test")
a = ap.parse_args()
tr = [json.loads(line) for line in (R / a.run / "trace.jsonl").open(encoding="utf-8")]
qs = [json.loads(line) for line in (R / "data" / a.split / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
results = {}
for t in tr:
    ans, conf = p.vote([s["pred"] for s in t["samples"]], len(t["samples"]))
    results[t["id"]] = t["final"] if t["final"].get("reason") else ans if p.decide(ans, conf, a.threshold) else {"type": "abstain"}
bad = [i for i, r in results.items() if problem({"id": i, **r})]  # e.g. a list with a NULL item: abstain, never a rejected row
for i in bad:
    results[i] = {"type": "abstain"}
if bad:
    print(f"{len(bad)} answers fail the format check, abstained: {bad}")
(R / a.out).mkdir(parents=True, exist_ok=True)
p.write_submission(R / a.out / "submission.jsonl", qs, results)
old = {r["id"]: r for r in map(json.loads, (R / a.run / "submission.jsonl").open(encoding="utf-8"))}
new = {r["id"]: r for r in map(json.loads, (R / a.out / "submission.jsonl").open(encoding="utf-8"))}
changed = [i for i in old if old[i] != new[i]]
print(f"threshold {a.threshold}: answered {sum(r['type'] != 'abstain' for r in old.values())} -> "
      f"{sum(r['type'] != 'abstain' for r in new.values())}, {len(changed)} rows changed -> {R / a.out / 'submission.jsonl'}")
