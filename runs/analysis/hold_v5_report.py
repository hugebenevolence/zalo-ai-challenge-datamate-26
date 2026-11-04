"""Score the v5 holdout run against train answers, compare with the v4 run on the same 150 questions."""
import glob, json, os, sys
from collections import Counter
REPO = os.path.expanduser("~/projects/zalo-ai-challenge-datamate-26")
sys.path.insert(0, f"{REPO}/src")
from datamate.scoring import score_one
refs = {r["id"]: r for r in map(json.loads, open(f"{REPO}/data/train/answers.jsonl"))}
def load(run):
    return {json.load(open(p))["id"]: json.load(open(p)) for p in glob.glob(f"{REPO}/runs/{run}/logs/*.json")}
v4, v5 = load("gemma_t2048_hold150"), load("gemma_v5_hold150")
for name, run in [("v4 (no dedupe)", v4), ("v5 (current baseline)", v5)]:
    s = [score_one(lg["final"], refs[q]) for q, lg in run.items()]
    c = Counter(s)
    print(f"{name}: N={len(s)} PA {sum(s)/len(s):+.3f} | right {c[1]} wrong {c[-1]} abstain {c[0]} | precision {c[1]/max(c[1]+c[-1],1):.2f}")
print("\nper shop (v4 -> v5):")
for b in sorted({lg["business_id"] for lg in v5.values()}):
    a = [score_one(v4[q]["final"], refs[q]) for q, lg in v5.items() if lg["business_id"] == b and q in v4]
    c = [score_one(lg["final"], refs[q]) for q, lg in v5.items() if lg["business_id"] == b]
    print(f"  {b}: {sum(a)/len(a):+.2f} -> {sum(c)/len(c):+.2f}")
print("\nv5 WRONG answers:")
for q, lg in v5.items():
    if score_one(lg["final"], refs[q]) == -1:
        print(f"  {lg['business_id']} sup {lg['support']} | {lg['question'][:80]} | pred {str(lg['final'].get('answer'))[:35]} | ref {str(refs[q]['answer'])[:35]}")
print("\nv5 ABSTAIN with at least one correct sample:")
for q, lg in v5.items():
    if lg["final"]["type"] == "abstain":
        ok = sum(score_one(s["answer"], refs[q]) == 1 for s in lg["samples"] if s.get("answer"))
        if ok: print(f"  {lg['business_id']} sup {lg['support']} correct samples {ok}/5 | {lg['question'][:80]}")
