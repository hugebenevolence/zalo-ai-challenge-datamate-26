"""How often do samples agree on a value but split on its type label (number vs percent)? Re-vote merging them.
python runs/analysis/vote_split.py RUN [--gold data/x/answers.jsonl]"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
from evaluation import score
run = R / sys.argv[1]
gold = {}
if "--gold" in sys.argv:
    gold = {x["id"]: x for x in map(json.loads, open(R / sys.argv[sys.argv.index("--gold") + 1]))}
dump = lambda p: json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
def vkey(p):  # number and percent with the same value count as one answer
    return dump({"type": "num", "answer": round(float(p["answer"]), 2)}) if p["type"] in ("number", "percent") and isinstance(p.get("answer"), (int, float)) and not isinstance(p["answer"], bool) else dump(p)
flips, stats = [], Counter()
for t in map(json.loads, open(run / "trace.jsonl")):
    preds = [s["pred"] for s in t["samples"] if s["pred"]]
    k = len(t["samples"])
    groups = defaultdict(list)
    for p in preds:
        groups[vkey(p)].append(p)
    if not groups: continue
    best = max(groups.values(), key=len)
    types = Counter(p["type"] for p in best)
    split = len(types) > 1
    stats["type split in best group"] += split
    old = t["final"]
    if split:
        stats["split & old abstain"] += old["type"] == "abstain"
        newconf = len(best) / k
        stats["split & merged conf>=0.5"] += newconf >= 0.5
        if old["type"] == "abstain" and newconf >= 0.5 and best[0].get("answer") not in (0, None):
            flips.append((t["id"], types, best[0]["answer"], newconf))
            if gold:
                g = gold[t["id"]]
                stats[f"flip gold type {g['type']}"] += 1
                for ty in types:
                    stats[f"flip {ty} would score {score({'type': ty, 'answer': best[0]['answer']}, g)}"] += 1
        if gold and old["type"] != "abstain":
            stats[f"answered-split old score {score(old, gold[t['id']])}"] += 1
print(dict(stats))
print(len(flips), "abstain -> answer flips with merged vote >= 0.5")
for f in flips[:15]: print(f)
