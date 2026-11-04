"""Learned answer/abstain gate: logistic regression on how the samples of one question agree, fitted on labeled train runs of
shops the LoRA never saw. python runs/analysis/gate.py fit RUN... --out sqlbase/gate_v1.json | apply RUN BASE_THRESHOLD --gate F --th 0.6 --out DIR [--split public_test]"""
import json, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "train")]
import numpy as np
import predict as P
from check_submission import problem
from evaluation import score
TYPES = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
NAMES = ["share", "second", "distinct", "failed", "A", "B", "think", "hard", "repaired", "zero", "negative", "listtable", "qlen"] + TYPES
def features(t):
    ps = [s["pred"] for s in t["samples"]]; a, c = P.vote(ps, len(ps))
    if a["type"] == "abstain":
        return a, None
    k = lambda p: json.dumps(p, sort_keys=True, ensure_ascii=False, default=str)
    top = Counter(k(p) for p in ps if p).most_common(2)
    same = lambda p: bool(p) and p.get("answer") == a.get("answer")
    ans = a.get("answer"); num = isinstance(ans, (int, float)) and not isinstance(ans, bool)
    return a, [c, top[1][1] / len(ps) if len(top) > 1 else 0, len(set(k(p) for p in ps if p)) / len(ps), sum(p is None for p in ps) / len(ps),
               sum(map(same, ps[:4])) / 4, sum(map(same, ps[4:8])) / 4, sum(map(same, ps[8:])) / max(len(ps[8:]), 1), float(len(ps) > 8),
               sum(1 for s in t["samples"] if s.get("error") and same(s["pred"])) / max(sum(map(same, ps)), 1), float(num and ans == 0), float(num and ans < 0),
               float(a["type"] in ("list", "table")), len(t["question"]) / 100] + [float(a["type"] == x) for x in TYPES]
def traces(runs):
    tr = {}
    for r in runs:
        for line in open(R / r / "trace.jsonl", encoding="utf-8"):
            t = json.loads(line); tr[t["id"]] = t  # a re-run shop appears twice: the later row wins
    return tr
if sys.argv[1] == "fit":
    from sklearn.linear_model import LogisticRegression
    i = sys.argv.index("--out"); out = R / sys.argv[i + 1]; runs = sys.argv[2:i]
    gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
    X, y = [], []
    for t in traces(runs).values():
        a, f = features(t)
        if f and not t["final"].get("reason"):
            X.append(f); y.append(int(score(a, gold[t["id"]]) == 1))
    X, y = np.array(X), np.array(y)
    m = LogisticRegression(max_iter=2000).fit(X, y)
    out.write_text(json.dumps({"names": NAMES, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "n": len(y)}, indent=1))
    print(f"fitted on {len(y)} questions, {y.mean():.1%} right ->", out)
    for n_, c in sorted(zip(NAMES, m.coef_[0]), key=lambda x: -abs(x[1]))[:8]: print(f"   {n_:10} {c:+.2f}")
else:
    args = sys.argv[2:]; run = args[0]; g = json.load(open(R / args[args.index("--gate") + 1])); th = float(args[args.index("--th") + 1]); out = R / args[args.index("--out") + 1]
    split = args[args.index("--split") + 1] if "--split" in args else "public_test"
    qs = [json.loads(l) for l in open(R / "data" / split / "questions.jsonl", encoding="utf-8-sig") if l.strip()]
    coef = np.array(g["coef"]); res = {}; pr = []
    for i, t in traces([run]).items():
        if t["final"].get("reason"):
            res[i] = t["final"]; continue
        a, f = features(t)
        p = float(1 / (1 + np.exp(-(coef @ np.array(f) + g["intercept"])))) if f else 0
        pr.append(p)
        r = a if f and p >= th and P.decide(a, 1.0, 0) else {"type": "abstain"}
        res[i] = {"type": "abstain"} if problem({"id": i, **r}) else r
    out.mkdir(parents=True, exist_ok=True); P.write_submission(out / "submission.jsonl", qs, res)
    print(f"{run}: gate {th}: answered {sum(r['type'] != 'abstain' for r in res.values())}, mean P(right) on all {np.mean(pr):.2f} -> {out / 'submission.jsonl'}")
