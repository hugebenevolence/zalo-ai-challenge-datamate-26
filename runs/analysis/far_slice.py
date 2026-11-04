# Holdout scored by how close each question is to the train questions the adapters saw, and reweighted to the public mix. Public questions are used only for that mix.
import json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P, fewshot
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
trq = {q["id"]: q for q in rd(R / "data/train/questions.jsonl")}; pub = rd(R / "data/public_test/questions.jsonl")
ids = [i.strip() for i in open(R / "splits/holdout600.txt") if i.strip()]
tr24 = [trq[i.strip()] for i in open(R / "splits/rft_train24.txt") if i.strip()]
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
sim = {i: nn(trq[i]["question"]) for i in ids}
edges = [0, .5, .6, .7, .8, .9, 1.01]; bk = lambda x: next(k for k in range(len(edges) - 1) if edges[k] <= x < edges[k + 1])
mix = Counter(bk(nn(q["question"])) for q in pub)
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
M = re.compile(r"debt_(at|new|repaid)\s*\(")
use = lambda s: bool(M.search(s["text"].split("\n#REPAIR\n")[-1]))
def sc(i, t, ss, route_from=None):
    if t["final"].get("reason"): return score_one({"id": i, **t["final"]}, gold[i])
    ps = [s["pred"] for s in ss]
    if route_from is not None:
        th = ss[route_from:]
        if th and sum(map(use, th)) / len(th) >= .5: ps = [s["pred"] for s in ss if use(s)]
    a, c = P.vote(ps, len(ps)); r = a if P.decide(a, c, .5) else {"type": "abstain"}
    return score_one({"id": i, **r}, gold[i])
base = {**load("runs/sql_q35_hold150"), **load("runs/sql_q35_base_rest450")}; l1 = {**load("runs/sql_q35_lora1_hold150"), **load("runs/sql_q35_lora1_rest450")}
v2, to, v3 = load("runs/sql_v2_holdout632"), load("runs/sql_v2_thinkonly632"), load("runs/sql_v3_prompt632")
print("coverage base", sum(i in base for i in ids), "lora1", sum(i in l1 for i in ids))
S = {"base 9B": {i: sc(i, base[i], base[i]["samples"]) for i in ids if i in base},
     "LoRA v1": {i: sc(i, l1[i], l1[i]["samples"]) for i in ids if i in l1},
     "LoRA v2, 8 plain only": {i: sc(i, v2[i], v2[i]["samples"][:8]) for i in ids},
     "LoRA v2 current, routing": {i: sc(i, v2[i], v2[i]["samples"], 8) for i in ids},
     "think-only K=4, routing": {i: sc(i, to[i], to[i]["samples"][2:6], 0) for i in ids},
     "think-only K=6, routing": {i: sc(i, to[i], to[i]["samples"][2:8], 0) for i in ids},
     "think-only K=8, routing": {i: sc(i, to[i], to[i]["samples"][2:10], 0) for i in ids},
     "prompt v3 K=4, routing": {i: sc(i, v3[i], v3[i]["samples"][2:6], 0) for i in ids}}
far = [i for i in ids if sim[i] < .7]; mid = [i for i in ids if .7 <= sim[i] < .9]; near = [i for i in ids if sim[i] >= .9]
print(f"holdout: far(<.7) {len(far)}, mid {len(mid)}, near(>=.9) {len(near)} | public mix far {sum(mix[k] for k in (0,1,2))/1000:.1%} mid {sum(mix[k] for k in (3,4))/1000:.1%} near {mix[5]/1000:.1%}")
pa = lambda r, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([r[i] for i in xs if i in r])
def wt(r):
    out = 0
    for k in range(6):
        xs = [i for i in ids if bk(sim[i]) == k]; out += mix[k] / 1000 * pa(r, xs)
    return out
print(f"{'system':28s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'public-mix':>10s}")
for n, r in S.items(): print(f"{n:28s} {pa(r,ids):6.3f} {pa(r,near):6.3f} {pa(r,mid):6.3f} {pa(r,far):6.3f} {wt(r):10.3f}")
c = S["LoRA v2 current, routing"]; random.seed(0)
def boot(fn, a, xs):
    out = []
    for _ in range(3000):
        smp = random.choices(xs, k=len(xs)); out.append(fn(a, smp))
    out.sort(); return out[75], out[2924]
for n in ("think-only K=6, routing", "think-only K=8, routing", "prompt v3 K=4, routing", "LoRA v2, 8 plain only"):
    r = S[n]; lo, hi = boot(lambda a, s: sum(a[i] - c[i] for i in s) / len(s), r, far)
    print(f"far slice {n} - current: {pa(r,far)-pa(c,far):+.3f} CI95 [{lo:+.3f}, {hi:+.3f}]")
for n in ("LoRA v1", "base 9B"):
    r = S[n]; xs = [i for i in far if i in r]; lo, hi = boot(lambda a, s: sum(c[i] - a[i] for i in s) / len(s), r, xs)
    print(f"far slice current - {n}: {pa(c,xs)-pa(r,xs):+.3f} CI95 [{lo:+.3f}, {hi:+.3f}]")
# public-mix differences with a bootstrap inside each similarity bucket
buckets = [[i for i in ids if bk(sim[i]) == k] for k in range(6)]
def mixdiff(a, b):
    out = []
    for _ in range(2000):
        t = 0
        for k, xs in enumerate(buckets):
            s = random.choices(xs, k=len(xs)); t += mix[k] / 1000 * sum(a[i] - b[i] for i in s) / len(s)
        out.append(t)
    out.sort(); return sum(sum(a[i]-b[i] for i in xs)/len(xs)*mix[k]/1000 for k, xs in enumerate(buckets)), out[50], out[1949]
for x, y in (("think-only K=6, routing", "LoRA v2 current, routing"), ("think-only K=8, routing", "LoRA v2 current, routing"), ("prompt v3 K=4, routing", "think-only K=4, routing"), ("prompt v3 K=4, routing", "LoRA v2 current, routing")):
    d, lo, hi = mixdiff(S[x], S[y]); print(f"public-mix {x} - {y}: {d:+.3f} CI95 [{lo:+.3f}, {hi:+.3f}]")
a, b = S["prompt v3 K=4, routing"], S["think-only K=4, routing"]; lo, hi = boot(lambda r, s: sum(r[i] - b[i] for i in s) / len(s), a, far)
print(f"far slice v3 K=4 - old K=4: {pa(a,far)-pa(b,far):+.3f} CI95 [{lo:+.3f}, {hi:+.3f}]")
