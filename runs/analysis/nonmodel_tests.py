import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path[:0] = ["sqlbase"]
import predict as P
from evaluation import score
from ingest import norm
gold = {x["id"]: x for x in map(json.loads, open("data/train/answers.jsonl"))}
def load(*runs):
    tr = {}
    for r in runs:
        for l in open(f"{r}/trace.jsonl"):
            t = json.loads(l); tr[t["id"]] = t
    return tr
DEBT = r"(^|_)(no|cong_no|con_no|tra_no|chiu|ghi_no|tuoi_no|nowj)(_|$)"
# 1. shops whose README describes no columns, debt questions left out (they have their own fix)
print("== 1. README with / without column descriptions (no LoRA, no thinking, 24 shops, debt questions excluded)")
tr = load("runs/sql_q35_rft_train24")
shop = defaultdict(Counter)
for t in tr.values():
    if re.search(DEBT, norm(t["question"])): continue
    ps = [s["pred"] for s in t["samples"]]; a, c = P.vote(ps, len(ps))
    r = t["final"] if t["final"].get("reason") else (a if P.decide(a, c, .5) else {"type": "abstain"})
    shop[t["business_id"]][score(r, gold[t["id"]])] += 1
def cols(b):
    txt = Path(f"data/train/data/{b}/README.md").read_text(encoding="utf-8-sig")
    return len(re.findall(r"(?m)^\s*-\s*\*\*[^*]+\*\*\s*:", txt.split("## Quy ước")[0]))
rows = sorted(((cols(b), b, v) for b, v in shop.items()), key=lambda x: x[0])
g = defaultdict(Counter)
for n, b, v in rows:
    g["no column list" if n == 0 else "column list"].update(v)
    print(f"   {b} described columns {n:3}  PA {(v[1]-v[-1])/sum(v.values()):.2f}  right/wrong/abstain {v[1]}/{v[-1]}/{v[0]}")
for k, v in g.items(): print(f"  {k:16} PA {(v[1]-v[-1])/sum(v.values()):.3f}  ({sum(v.values())} q, right {v[1]} wrong {v[-1]} abstain {v[0]})")
# 2. position-wise vote for lists
print("== 2. list answers: vote per position instead of on the whole list (LoRA, 632 q)")
tr = load("runs/sql_q35_lora1_hold150", "runs/sql_q35_lora1_rest450")
def pos_vote(ps):
    ls = [p["answer"] for p in ps if p and p["type"] == "list" and isinstance(p.get("answer"), list) and p["answer"]]
    if not ls: return None, 0
    n, cn = Counter(len(x) for x in ls).most_common(1)[0]
    same = [x for x in ls if len(x) == n]
    out, shares = [], []
    for i in range(n):
        v, c = Counter(json.dumps(x[i], ensure_ascii=False) for x in same).most_common(1)[0]
        out.append(json.loads(v)); shares.append(c / max(len(ps), 1))
    return {"type": "list", "answer": out}, min(shares)
for th in (.5, .4, .34, .25):
    a_ = Counter(); b_ = Counter()
    for t in tr.values():
        ps = [s["pred"] for s in t["samples"]]; a, c = P.vote(ps, len(ps)); g_ = gold[t["id"]]
        cur = a if P.decide(a, c, .5) else {"type": "abstain"}
        new = cur
        if cur["type"] == "abstain" or a["type"] == "list":
            ls = sum(1 for p in ps if p and p["type"] == "list")
            if ls >= max(1, len(ps) / 2):
                x, sh = pos_vote(ps)
                new = x if x and sh >= th else {"type": "abstain"}
        if t["final"].get("reason"): cur = new = t["final"]
        a_[score(cur, g_)] += 1; b_[score(new, g_)] += 1
    print(f"   position share >= {th}: today PA {(a_[1]-a_[-1])/len(tr):.3f} ({a_[1]}/{a_[-1]}/{a_[0]})  ->  per position PA {(b_[1]-b_[-1])/len(tr):.3f} ({b_[1]}/{b_[-1]}/{b_[0]})")
# 3. learned gate, leave one shop out
print("== 3. learned answer/abstain gate, leave-one-shop-out (LoRA, 632 q)")
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    print("   sklearn missing:", e); sys.exit()
X, y, ids, biz = [], [], [], []
TYPES = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
for t in tr.values():
    ps = [s["pred"] for s in t["samples"]]; a, c = P.vote(ps, len(ps))
    if a["type"] == "abstain" or t["final"].get("reason"): continue
    k = lambda p: json.dumps(p, sort_keys=True, ensure_ascii=False, default=str)
    cnt = Counter(k(p) for p in ps if p); top = cnt.most_common(2); sec = top[1][1] / len(ps) if len(top) > 1 else 0
    same = lambda p: p and p.get("answer") == a.get("answer")
    A, B, Th = sum(1 for p in ps[:4] if same(p)) / 4, sum(1 for p in ps[4:8] if same(p)) / 4, (sum(1 for p in ps[8:] if same(p)) / max(len(ps[8:]), 1))
    rep = sum(1 for s in t["samples"] if s.get("error") and same(s["pred"])) / max(sum(1 for p in ps if same(p)), 1)
    ans = a.get("answer"); num = isinstance(ans, (int, float)) and not isinstance(ans, bool)
    X.append([c, sec, len(cnt) / len(ps), sum(1 for p in ps if p is None) / len(ps), A, B, Th, float(len(ps) > 8), rep, float(num and ans == 0), float(num and ans < 0), float(a["type"] in ("list", "table")), len(t["question"]) / 100] + [float(a["type"] == x) for x in TYPES])
    y.append(int(score(a, gold[t["id"]]) == 1)); ids.append(t["id"]); biz.append(t["business_id"])
import numpy as np
X, y, biz = np.array(X), np.array(y), np.array(biz)
n = len(tr)
for name, mk in [("logistic", lambda: LogisticRegression(max_iter=2000)), ("boosted trees", lambda: GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=.05, random_state=0))]:
    pr = np.zeros(len(y))
    for b in set(biz):
        m = biz == b
        pr[m] = mk().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
    for th in (.5, .6, .7):
        ans = pr >= th; right, wrong = int(y[ans].sum()), int((1 - y[ans]).sum())
        extra = sum(1 for t in tr.values() if t["final"].get("reason") and score(t["final"], gold[t["id"]]) == 1)
        print(f"   {name:13} P(right) >= {th}: PA {(right + extra - wrong)/n:.3f} (right {right + extra} wrong {wrong})")
c = Counter(); 
for t in tr.values():
    c[score(t["final"], gold[t["id"]])] += 1
print(f"   today's rule (share >= 0.5): PA {(c[1]-c[-1])/n:.3f} (right {c[1]} wrong {c[-1]})")
