"""Few-shot bank from a labeled run: python runs/analysis/build_fewshot_bank.py RUN [RUN ...] --out sqlbase/fewshot_bank.jsonl
For each question one reply whose answer matches gold and whose SQL reads only the canonical views (the shortest, up to 900 characters)."""
import argparse, json, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import fewshot
from evaluation import score
from predict import parse
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--max-sql", type=int, default=900)
a = ap.parse_args()
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rows, types = [], Counter()
for run in a.runs:
    for line in open(R / run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line); g = gold[t["id"]]; best = None
        for s in t["samples"]:
            if s["pred"] and score(s["pred"], g) == 1:
                kind, sql = parse(s["text"].split("\n#REPAIR\n")[-1])
                if sql and kind == g["type"] and fewshot.tables_of(sql) and fewshot.tables_of(sql) <= fewshot.CANON and len(sql) <= a.max_sql and (best is None or len(sql) < len(best)):
                    best = sql
        if best:
            rows.append({"id": t["id"], "business_id": t["business_id"], "question": t["question"], "type": g["type"], "sql": best.rstrip(";")})
            types[g["type"]] += 1
Path(R / a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"{len(rows)} examples -> {a.out}; by type {dict(types.most_common())}; shops {len({r['business_id'] for r in rows})}")
