"""Per-stage statistics of one predict.py run (no labels needed): python runs/analysis/stage_stats.py RUN [--gold answers.jsonl]
Samples 0-3 use SYSTEM (views), 4-7 SYSTEM_ALT (raw tables, no views), 8+ thinking samples alternating A, B."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
from evaluation import score
run = R / sys.argv[1]
gold = {x["id"]: x for x in map(json.loads, open(R / sys.argv[sys.argv.index("--gold") + 1]))} if "--gold" in sys.argv else {}
dump = lambda p: json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)
def errkind(e):
    e = e.split(" | ")[-1]
    for k, pat in [("column/table not found", r"not found in FROM|does not have a column|Referenced column|Table with name|Catalog Error"),
                   ("empty / NULL result", r"result is NULL|got 0 rows"), ("wrong shape (rows/cols)", r"expected one cell|needs one column|got \d+ rows"),
                   ("no SQL / bad TYPE", r"need one read-only|TYPE line"), ("type/cast error", r"Conversion Error|Cannot compare|cast|Binder Error: No function"),
                   ("syntax", r"Parser Error|syntax"), ("timeout / memory", r"interrupt|Out of Memory|timeout"), ("list item NULL/nested", r"every list item"),
                   ("lint % change", r"a change in %")]:
        if re.search(pat, e, re.I): return k
    return "other"
c, errs, fixes = Counter(), Counter(), Counter()
per = {"A": Counter(), "B": Counter(), "thinkA": Counter(), "thinkB": Counter()}
for t in map(json.loads, open(run / "trace.jsonl")):
    s = t["samples"]
    c["questions"] += 1
    c["thinking triggered"] += len(s) > 8
    for j, x in enumerate(s):
        grp = "A" if j < 4 else "B" if j < 8 else ("thinkA" if (j - 8) % 2 == 0 else "thinkB")
        ok = x["pred"] is not None
        rep = bool(x.get("error"))
        per[grp]["n"] += 1; per[grp]["ok"] += ok; per[grp]["repaired ok"] += ok and rep; per[grp]["failed"] += not ok
        if gold and ok: per[grp]["correct"] += score(x["pred"], gold[t["id"]]) == 1
        if not ok: errs[errkind(x.get("error") or "")] += 1
        if rep: fixes["had error"] += 1; fixes["fixed by repair"] += ok
    # prompt A vs B majority
    A = Counter(dump(x["pred"]) for x in s[:4] if x["pred"]); B = Counter(dump(x["pred"]) for x in s[4:8] if x["pred"])
    if A and B:
        a, b = A.most_common(1)[0][0], B.most_common(1)[0][0]
        c["A,B both have answers"] += 1; c["A majority == B majority"] += a == b
        if gold:
            g = gold[t["id"]]
            c["A majority correct"] += score(json.loads(a), g) == 1; c["B majority correct"] += score(json.loads(b), g) == 1
    if gold:
        c["final correct"] += score(t["final"], gold[t["id"]]) == 1; c["final wrong"] += score(t["final"], gold[t["id"]]) == -1
        c["any sample correct"] += any(x["pred"] and score(x["pred"], gold[t["id"]]) == 1 for x in s)
print(json.dumps(dict(c), ensure_ascii=False))
for g, v in per.items():
    if v["n"]: print(g, dict(v), f"ok {v['ok']/v['n']:.0%}" + (f", correct {v['correct']/v['n']:.0%} of samples" if gold else ""))
print("failed samples by last error:", errs.most_common())
print("repairs:", dict(fixes))
