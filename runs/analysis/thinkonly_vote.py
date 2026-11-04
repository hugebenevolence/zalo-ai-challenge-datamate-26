"""Votes over thinking samples only, K = 2..8, on a labeled run where every question has them: python runs/analysis/thinkonly_vote.py RUN [--plain N]
--plain N: number of plain samples before the thinking samples (2 for --k 2)."""
import json, re, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
run = sys.argv[1]; start = int(sys.argv[sys.argv.index("--plain") + 1]) if "--plain" in sys.argv else 2
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / run / "trace.jsonl")}
n = len(tr); DEBT = re.compile(r"debt_(at|new|repaid)\s*\(")
total = max(len(t["samples"]) for t in tr.values()) - start
print(f"{n} questions, up to {total} thinking samples")
for k in [x for x in (2, 3, 4, 6, 8) if x <= total]:
    for th in (.5, .6, .75):
        c = Counter()
        for i, t in tr.items():
            if t["final"].get("reason"):
                c[score_one({"id": i, **t["final"]}, gold[i])] += 1; continue
            ps = [s["pred"] for s in t["samples"][start:start + k]]
            a, cf = P.vote(ps, len(ps))
            r = a if P.decide(a, cf, th) else {"type": "abstain"}
            c[score_one({"id": i, **r}, gold[i])] += 1
        print(f"  K={k} share>={th}: PA {(c[1]-c[-1])/n:.3f}  {c[1]}/{c[-1]}/{c[0]}")
