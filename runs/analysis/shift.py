# Diagnostic only: compares public QUESTIONS and our own outputs with the labeled holdout. Prints aggregates. Uses no public answers; nothing here feeds training.
import json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P, fewshot
from ingest import norm
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
trq = {q["id"]: q for q in rd(R / "data/train/questions.jsonl")}
pub = rd(R / "data/public_test/questions.jsonl")
hold = [trq[i.strip()] for i in open(R / "splits/holdout600.txt") if i.strip()]
tr24 = [trq[i.strip()] for i in open(R / "splits/rft_train24.txt") if i.strip()]
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
H, Pb = load("runs/sql_v2_holdout632"), load("runs/sql_v2_public")
print("train question keys", sorted(trq[hold[0]["id"]].keys()), "| public shops", len({q["business_id"] for q in pub}), "| holdout shops", len({q["business_id"] for q in hold}))
M = re.compile(r"debt_(at|new|repaid)\s*\(")
use = lambda s: bool(M.search(s["text"].split("\n#REPAIR\n")[-1]))
def final(t):
    if t["final"].get("reason"): return t["final"], 1.0, False
    ss = t["samples"]; th = ss[8:]; routed = bool(th) and sum(map(use, th)) / len(th) >= .5
    ps = [s["pred"] for s in ss if use(s)] if routed else [s["pred"] for s in ss]
    a, c = P.vote(ps, len(ps))
    return (a if P.decide(a, c, .5) else {"type": "abstain"}), c, routed
res = {}
for i, t in H.items():
    r, c, ro = final(t); res[i] = (score_one({"id": i, **r}, gold[i]), c, r["type"], ro)
pubres = {}
for i, t in Pb.items():
    r, c, ro = final(t); pubres[i] = (None, c, r["type"], ro)
n = len(res); cc = Counter(v[0] for v in res.values()); print(f"holdout PA {(cc[1]-cc[-1])/n:.3f} {cc[1]}/{cc[-1]}/{cc[0]}; public answered {sum(v[2]!='abstain' for v in pubres.values())}/1000 (leaderboard PA .491)")
# 1. rulebook
a = [l.strip() for l in open(R / "data/train/rulebook.md", encoding="utf-8") if l.strip()]; b = [l.strip() for l in open(R / "data/public_test/rulebook.md", encoding="utf-8") if l.strip()]
print(f"rulebook: train {sum(map(len,a))} chars {len(a)} lines; public {sum(map(len,b))} chars {len(b)} lines; public lines also in train {sum(l in set(a) for l in b)}/{len(b)}; identical file {a==b}")
# 2. nearest train question
F = [fewshot.feats(q["question"]) for q in tr24]; df = Counter(k for f in F for k in f); N = len(F)
idf = {k: math.log(1 + N / v) for k, v in df.items()}; nm = [math.sqrt(sum((c * idf[k]) ** 2 for k, c in f.items())) for f in F]
inv = defaultdict(list)
for j, f in enumerate(F):
    for k, c in f.items(): inv[k].append((j, c))
def nn(text):
    f = fewshot.feats(text); nq = math.sqrt(sum((c * idf.get(k, 1.0)) ** 2 for k, c in f.items())) or 1.0
    s = defaultdict(float)
    for k, c in f.items():
        for j, g in inv.get(k, ()): s[j] += c * g * idf[k] ** 2
    return max((v / (nq * nm[j]) for j, v in s.items()), default=0.0)
hs = {q["id"]: nn(q["question"]) for q in hold}; ps = {q["id"]: nn(q["question"]) for q in pub}
edges = [0, .5, .6, .7, .8, .9, 1.01]
def bk(x): return next(k for k in range(len(edges) - 1) if edges[k] <= x < edges[k + 1])
hb, pbk = defaultdict(list), Counter(bk(v) for v in ps.values())
for i, v in hs.items(): hb[bk(v)].append(res[i][0])
print("nearest train question similarity: bucket | holdout share PA | public share")
exp = 0
for k in range(len(edges) - 1):
    x = hb[k]; pa = (x.count(1) - x.count(-1)) / len(x) if x else 0; exp += pbk[k] / 1000 * pa
    print(f"  {edges[k]:.1f}-{edges[k+1]:.1f}: {len(x)/n:5.1%} PA {pa:+.3f} | {pbk[k]/1000:5.1%}")
print(f"holdout PA reweighted to the public similarity mix: {exp:.3f}")
# 3. keyword domains
D = [("debt", r"\bno\b|cong_no|thu_no|con_no|tra_no"), ("delivery", r"giao_hang|van_chuyen|ship|giao_tre|tre_han|giao_dung|don_vi_giao"), ("inventory", r"ton_kho|\bton\b|nhap_kho|xuat_kho|kiem_kho|het_hang"),
     ("purchase/cost", r"nhap_hang|nha_cung_cap|ncc|gia_von|chi_phi|loi_nhuan|bien_loi|lai_gop|phieu_nhap"), ("price change", r"doi_gia|tang_gia|giam_gia|thay_doi_gia|gia_ban|bang_gia"), ("shift/channel", r"\bca\b|kenh|chi_nhanh|nhan_vien|ca_lam"), ("customer", r"khach")]
def dom(text):
    t = "_" + norm(text) + "_"; t2 = t.replace("_", " ").strip()
    t3 = norm(text)
    for n_, pat in D:
        if re.search(pat.replace(r"\b", r"(?:^|_)", 1).replace(r"\b", r"(?:_|$)") if r"\b" in pat else pat, t3): return n_
    return "sales/other"
hd, pdm = defaultdict(list), defaultdict(list)
for q in hold: hd[dom(q["question"])].append(res[q["id"]][0])
for q in pub: pdm[dom(q["question"])].append(pubres[q["id"]])
print("domain: holdout share PA abstain | public share abstain mean-agreement")
exp = 0
for k in sorted(set(hd) | set(pdm), key=lambda z: -len(pdm[z])):
    x = hd[k]; pa = (x.count(1) - x.count(-1)) / len(x) if x else 0; y = pdm[k]; exp += len(y) / 1000 * pa
    print(f"  {k:14s} {len(x)/n:5.1%} PA {pa:+.3f} abst {x.count(0)/max(1,len(x)):4.0%} | {len(y)/1000:5.1%} abst {sum(v[2]=='abstain' for v in y)/max(1,len(y)):4.0%} agree {sum(v[1] for v in y)/max(1,len(y)):.2f}")
print(f"holdout PA reweighted to the public domain mix: {exp:.3f}")
# 4. agreement -> precision
ed = [0, .5, .6, .75, .9, 1.01]; bb = lambda x: next(k for k in range(len(ed) - 1) if ed[k] <= x < ed[k + 1])
h2, p2 = defaultdict(list), Counter()
for v in res.values():
    if v[2] != "abstain": h2[bb(v[1])].append(v[0])
for v in pubres.values():
    if v[2] != "abstain": p2[bb(v[1])] += 1
ansh = sum(map(len, h2.values())); ansp = sum(p2.values()); exp = 0
print("answered questions by vote agreement: bucket | holdout share precision | public share")
for k in range(len(ed) - 1):
    x = h2[k]; pa = (x.count(1) - x.count(-1)) / len(x) if x else 0; exp += p2[k] * pa
    print(f"  {ed[k]:.2f}-{ed[k+1]:.2f}: {len(x)/ansh:5.1%} right {x.count(1)/max(1,len(x)):.3f} | {p2[k]/ansp:5.1%}")
print(f"expected public PA if holdout precision per agreement bucket held: {exp/1000:.3f} (leaderboard .491; answered {ansp})")
# 5. run behaviour
def beh(tr):
    ss = [s for t in tr.values() for s in t["samples"]]
    canon = lambda s: (lambda tb: bool(tb) and tb <= fewshot.CANON)(fewshot.tables_of(s["text"].split("\n#REPAIR\n")[-1].split("</think>")[-1]))
    return dict(q=len(tr), hard=sum(len(t["samples"]) > 8 for t in tr.values()), samples=len(ss), none=sum(not s["pred"] for s in ss) / len(ss), err=sum(bool(s["error"]) for s in ss) / len(ss), canon=sum(map(canon, ss)) / len(ss), unans=sum((s["pred"] or {}).get("type") == "unanswerable" for s in ss) / len(ss))
for n_, tr in (("holdout", H), ("public", Pb)):
    b_ = beh(tr); print(n_, {k: round(v, 3) for k, v in b_.items()})
# 6. per public shop
sh = defaultdict(list)
for i, t in Pb.items(): sh[t["business_id"]].append(pubres[i])
print("public shop: answered share, mean agreement, routed")
for k, v in sorted(sh.items()):
    print(f"  {k}: n={len(v)} answered {sum(x[2]!='abstain' for x in v)/len(v):.2f} agree {sum(x[1] for x in v)/len(v):.2f} routed {sum(x[3] for x in v)}")
