"""Offline vote variants that lean on the thinking samples of hard questions (share < 0.75 on the K plain samples):
python runs/analysis/think_vote.py RUN [--gold answers.jsonl] [--write OUT --variant NAME --split public_test]"""
import argparse, importlib.util, json, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "train")]
spec = importlib.util.spec_from_file_location("predict", R / "sqlbase/predict.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
from evaluation import score
from check_submission import problem
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--gold"); ap.add_argument("--write"); ap.add_argument("--variant"); ap.add_argument("--split", default="public_test")
a = ap.parse_args()
gold = {x["id"]: x for x in map(json.loads, open(R / a.gold))} if a.gold else {}
tr = {}
for line in open(R / a.run / "trace.jsonl"):
    t = json.loads(line); tr[t["id"]] = t  # a re-run shop appears twice: the later row wins
def decide_v(t, v):
    ps = [s["pred"] for s in t["samples"]]
    if t["final"].get("reason"):
        return t["final"]
    plain, think = ps[:8], ps[8:]
    if v.startswith("all@"):                      # today's rule: one vote over all samples
        ans, c = P.vote(ps, len(ps)); th = float(v[4:])
    elif v.startswith("w2@"):                     # a thinking sample counts twice
        q = plain + think + think; ans, c = P.vote(q, len(q)); th = float(v[3:])
    elif v.startswith("think@"):                  # hard question: thinking samples alone decide
        if think:
            ans, c = P.vote(think, len(think))
        else:
            ans, c = P.vote(plain, len(plain))
        th = float(v[6:])
    elif v.startswith("think+@"):                 # hard question: thinking alone at >= th, else the all-sample vote at 0.5
        th = float(v[7:])
        if think:
            ans, c = P.vote(think, len(think))
            if not P.decide(ans, c, th):
                ans, c = P.vote(ps, len(ps)); th = 0.5
        else:
            ans, c = P.vote(plain, len(plain))
    return ans if P.decide(ans, c, th) else {"type": "abstain"}
variants = ["all@0.5", "all@0.4", "w2@0.5", "w2@0.4", "w2@0.34", "think@0.5", "think@0.75", "think+@0.5", "think+@0.75"]
if gold:
    for v in variants:
        c = Counter()
        for i, t in tr.items():
            s = score(decide_v(t, v), gold[i]); c[s] += 1
        print(f"{v:12} PA {(c[1]-c[-1])/len(tr):.3f}  right {c[1]} wrong {c[-1]} abstain {c[0]}")
else:
    base = {i: decide_v(t, "all@0.4") for i, t in tr.items()}
    for v in variants:
        d = {i: decide_v(t, v) for i, t in tr.items()}
        print(f"{v:12} answered {sum(r['type'] != 'abstain' for r in d.values())}, differs from all@0.4 on {sum(d[i] != base[i] for i in d)}")
if a.write:
    qs = [json.loads(l) for l in open(R / "data" / a.split / "questions.jsonl", encoding="utf-8-sig") if l.strip()]
    res = {}
    for i, t in tr.items():
        r = decide_v(t, a.variant); res[i] = {"type": "abstain"} if problem({"id": i, **r}) else r
    (R / a.write).mkdir(parents=True, exist_ok=True)
    P.write_submission(R / a.write / "submission.jsonl", qs, res)
    print("wrote", a.write, a.variant, "answered", sum(r["type"] != "abstain" for r in res.values()))
