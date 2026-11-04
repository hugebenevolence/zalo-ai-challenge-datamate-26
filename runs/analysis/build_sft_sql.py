"""SFT rows for the SQL generator from a labeled predict.py run (--save-prompts): replies whose executed answer matches
gold, prompted exactly as at inference. python runs/analysis/build_sft_sql.py RUN DATA OUT.jsonl [--max N] [--per-q 1]
Samples 0..3 were asked with SYSTEM, 4..7 with SYSTEM_ALT (K=8); a repaired sample contributes its fix as the reply;
thinking samples are left out (the adapter serves the non-thinking samples)."""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
spec = importlib.util.spec_from_file_location("predict", R / "sqlbase/predict.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
from evaluation import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("data", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--max", type=int, default=1800)
ap.add_argument("--per-q", type=int, default=1)
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
rulebook = (R / a.data / "rulebook.md").read_text(encoding="utf-8-sig")
systems = [s + "\n\nRULEBOOK:\n" + rulebook for s in (P.SYSTEM, P.SYSTEM_ALT)]
gold = {x["id"]: x for x in map(json.loads, open(R / a.data / "answers.jsonl", encoding="utf-8"))}
rng = random.Random(0)
by_q, stats = [], {"questions": 0, "with a right sample": 0}
for t in map(json.loads, open(R / a.run / "trace.jsonl", encoding="utf-8")):
    stats["questions"] += 1
    if "prompt_user" not in t:
        continue
    good, seen = [], set()
    for j, smp in enumerate(t["samples"][:a.k]):
        if smp["pred"] and score(smp["pred"], gold[t["id"]]) == 1:
            reply = smp["text"].split("\n#REPAIR\n")[-1].strip()
            key = reply.split("```sql")[-1]
            if key not in seen:
                seen.add(key)
                good.append({"system": systems[j // (a.k // 2)], "user": t["prompt_user"], "reply": reply, "id": t["id"]})
    if good:
        stats["with a right sample"] += 1
        rng.shuffle(good)
        by_q.append(good[:a.per_q])
rng.shuffle(by_q)
rows = [r for g in by_q for r in g][:a.max]
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{stats}; {len(rows)} SFT rows -> {a.out}")
