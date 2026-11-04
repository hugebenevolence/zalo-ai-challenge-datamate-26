"""Debt questions on shops with the debt layer: plain samples of LoRA v2 mostly ignore the macros (it never saw them), thinking samples use
them. Offline: when most thinking samples call a debt macro, vote only among the samples that do. python runs/analysis/debt_vote.py RUN [--gold] [--write OUT]"""
import json, re, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "train"), str(R / "src")]
import predict as P
from check_submission import problem
from evaluation import score
MACRO = re.compile(r"debt_(at|new|repaid)\s*\(")
def decide(t, th=.5, need=.5):
    ps = [s["pred"] for s in t["samples"]]
    if t["final"].get("reason"):
        return t["final"], False
    uses = [bool(MACRO.search(s["text"].split("\n#REPAIR\n")[-1])) for s in t["samples"]]
    think = uses[8:]
    routed = bool(think) and sum(think) / len(think) >= need
    sub = [p for p, u in zip(ps, uses) if u] if routed else ps
    a, c = P.vote(sub, len(sub))
    return (a if P.decide(a, c, th) else {"type": "abstain"}), routed
tr = {}
for l in open(R / sys.argv[1] / "trace.jsonl", encoding="utf-8"):
    t = json.loads(l); tr[t["id"]] = t
if "--gold" in sys.argv:
    from datamate.scoring import score_one
    gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
    for th in (.5, .4):
        c, d, e, rc, rd = Counter(), Counter(), Counter(), Counter(), Counter()
        for i, t in tr.items():
            a, c0 = P.vote([s["pred"] for s in t["samples"]], len(t["samples"]))
            old = t["final"] if t["final"].get("reason") else (a if P.decide(a, c0, th) else {"type": "abstain"})
            new, routed = decide(t, th)
            c[score(old, gold[i])] += 1; d[score(new, gold[i])] += 1; e[score_one({"id": i, **new}, gold[i])] += 1
            if routed: rc[score(old, gold[i])] += 1; rd[score(new, gold[i])] += 1
        n = len(tr)
        print(f"threshold {th}: PA {(c[1]-c[-1])/n:.3f} {c[1]}/{c[-1]}/{c[0]} -> routed {(d[1]-d[-1])/n:.3f} {d[1]}/{d[-1]}/{d[0]} (other scorer {(e[1]-e[-1])/n:.3f}) | routed questions {sum(rd.values())}: {rc[1]}/{rc[-1]}/{rc[0]} -> {rd[1]}/{rd[-1]}/{rd[0]}")
if "--write" in sys.argv:
    out = R / sys.argv[sys.argv.index("--write") + 1]
    qs = [json.loads(l) for l in open(R / "data/public_test/questions.jsonl", encoding="utf-8-sig") if l.strip()]
    res, n = {}, 0
    for i, t in tr.items():
        r, routed = decide(t); n += routed
        res[i] = {"type": "abstain"} if problem({"id": i, **r}) else r
    out.mkdir(parents=True, exist_ok=True); P.write_submission(out / "submission.jsonl", qs, res)
    old = {r["id"]: r for r in map(json.loads, open(R / sys.argv[1] / "submission.jsonl"))}
    new = {r["id"]: r for r in map(json.loads, open(out / "submission.jsonl"))}
    ch = Counter("abstain->answer" if old[i]["type"] == "abstain" else "answer->abstain" if new[i]["type"] == "abstain" else "changed" for i in old if old[i] != new[i])
    print(f"public: {n} routed questions, {sum(ch.values())} rows changed {dict(ch)}, answered {sum(r['type'] != 'abstain' for r in new.values())} -> {out}")
