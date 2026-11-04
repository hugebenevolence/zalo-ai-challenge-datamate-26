"""Which parts of the pipeline earn points? Labeled runs only, no LLM: python runs/analysis/ablate.py RUN1 RUN2 ...
Re-votes subsets of the samples (A = views prompt, B = raw prompt, T = thinking) at several thresholds; loss buckets; by type and shop."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import predict as P
from evaluation import score
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
tr = {}
for run in sys.argv[1:]:
    for l in open(R / run / "trace.jsonl"):
        t = json.loads(l); tr[t["id"]] = t
n = len(tr)
def pa(fn):
    c = Counter()
    for t in tr.values():
        if t["final"].get("reason"):
            c[score(t["final"], gold[t["id"]])] += 1; continue
        ps = [s["pred"] for s in t["samples"]]
        sub, th = fn(ps)
        a, conf = P.vote(sub, len(sub)); r = a if P.decide(a, conf, th) else {"type": "abstain"}
        c[score(r, gold[t["id"]])] += 1
    return f"PA {(c[1]-c[-1])/n:.3f}  right {c[1]:3} wrong {c[-1]:3} abstain {c[0]:3} (precision {c[1]/max(c[1]+c[-1],1):.1%})"
print(f"{n} questions")
print("-- which samples matter")
for name, fn in [("A+B+thinking @0.5 (today)", lambda p: (p, .5)), ("A+B+thinking @0.4", lambda p: (p, .4)), ("A+B no thinking @0.5", lambda p: (p[:8], .5)), ("A+B no thinking @0.375", lambda p: (p[:8], .375)),
                 ("A only (4 samples) @0.5", lambda p: (p[:4], .5)), ("B only (4 samples) @0.5", lambda p: (p[4:8], .5)), ("A+thinking @0.5", lambda p: (p[:4] + p[8:], .5)), ("A+thinking @0.4", lambda p: (p[:4] + p[8:], .4))]:
    print(f"  {name:28} {pa(fn)}")
print("-- where the points are lost (today's vote @0.5)")
b = Counter(); ty = defaultdict(Counter); shop = defaultdict(Counter); stab = Counter()
for t in tr.values():
    g = gold[t["id"]]; ps = [s["pred"] for s in t["samples"]]
    a, conf = P.vote(ps, len(ps)); r = t["final"] if t["final"].get("reason") else (a if P.decide(a, conf, .5) else {"type": "abstain"})
    s = score(r, g); right = sum(1 for p in ps if p and score(p, g) == 1)
    k = "right" if s == 1 else ("wrong, right sample existed" if right else "wrong, no right sample") if s == -1 else ("abstain, right sample existed" if right else "abstain, no right sample")
    b[k] += 1; ty[g["type"]][s] += 1; shop[t["business_id"]][s] += 1
    stab["samples right share"] += right / max(len(ps), 1)
for k, v in b.most_common(): print(f"  {k:32} {v:4} ({v/n:.1%})")
print("  mean share of right samples:", round(stab["samples right share"] / n, 3))
print("-- by gold answer type   right/wrong/abstain")
for k, v in sorted(ty.items(), key=lambda x: -sum(x[1].values())): print(f"  {k:13} {v[1]:3}/{v[-1]:3}/{v[0]:3}  PA {(v[1]-v[-1])/sum(v.values()):.2f}")
print("-- by shop")
for k, v in sorted(shop.items()): print(f"  {k} {v[1]:3}/{v[-1]:3}/{v[0]:3}  PA {(v[1]-v[-1])/sum(v.values()):.2f}")
