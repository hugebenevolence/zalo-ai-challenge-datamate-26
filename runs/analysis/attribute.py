"""Where each labeled question is lost: python runs/analysis/attribute.py RUN answers.jsonl questions.jsonl
generation (no sample right) vs vote (a right sample existed) vs decide; per answer type; examples of losses."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
from evaluation import score
run, gold, qs = R / sys.argv[1], {x["id"]: x for x in map(json.loads, open(R / sys.argv[2]))}, {q["id"]: q for q in map(json.loads, open(R / sys.argv[3]))}
dump = lambda p: json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)
c, bytype, ex = Counter(), defaultdict(Counter), defaultdict(list)
for t in map(json.loads, open(run / "trace.jsonl")):
    g = gold[t["id"]]; f = t["final"]; s = [x["pred"] for x in t["samples"] if x["pred"]]
    sc = score(f, g); right = [p for p in s if score(p, g) == 1]
    share = len(right) / len(t["samples"]) if t["samples"] else 0
    if sc == 1: k = "right"
    elif sc == -1: k = "WRONG (a right sample existed)" if right else "WRONG (no sample right)"
    elif right: k = "abstain, right answer was in the vote" + (" (decide rule)" if share >= 0.5 else f" (share {share:.2f})")
    else: k = "abstain, no sample right"
    c[k] += 1; bytype[g["type"]][k.split(" (")[0].split(",")[0]] += 1
    if k != "right":
        top = Counter(dump(p) for p in s).most_common(3)
        ex[k].append(f'{t["id"]} [{g["type"]}] {qs[t["id"]]["question"][:100]}\n    gold={json.dumps(g.get("answer"), ensure_ascii=False)[:80]}  final={dump(f)[:80]}\n    top={[(a[:60], n) for a, n in top]}')
print(dict(c))
for ty, v in sorted(bytype.items()): print(f"  {ty:12} {dict(v)}")
for k, v in ex.items():
    print(f"\n=== {k}: {len(v)}")
    for x in v[:12]: print(x)
