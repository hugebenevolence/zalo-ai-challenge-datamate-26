"""SFT rows for LoRA v2, weighted to the questions the model still finds hard, from labeled runs with --save-prompts:
python runs/analysis/build_sft_v2.py --plain RUN --think RUN --out data/sft/sql_rft_v2.jsonl [--fewshot BANK] [--max 2400] [--easy 300]
Plain replies: every question with 1-5 of 8 right samples gives up to 2 distinct solutions, questions with 6-8 right give --easy rows
in all (so the adapter does not drift off them). Thinking replies that ran to the gold answer without a repair turn: 1 per question.
With --fewshot all rows use SYSTEM (no raw-tables prompt) and half of them carry examples from other shops, as at inference.
Half of the rows get the rulebook sections in another order, so that nothing is learned from where a rule sits."""
import argparse, importlib.util, json, random, re, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
spec = importlib.util.spec_from_file_location("predict", R / "sqlbase/predict.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import fewshot
from evaluation import score
ap = argparse.ArgumentParser()
ap.add_argument("--plain", type=Path, required=True); ap.add_argument("--think", type=Path); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fewshot", type=Path); ap.add_argument("--fewshot-k", type=int, default=4); ap.add_argument("--max", type=int, default=2400); ap.add_argument("--easy", type=int, default=300)
a = ap.parse_args()
rng = random.Random(0)
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rulebook = (R / "data/train/rulebook.md").read_text(encoding="utf-8-sig")
head, *sections = re.split(r"(?m)^(?=## )", rulebook)
def book():
    if rng.random() < 0.5:
        return rulebook
    s = sections[:]; rng.shuffle(s)
    return head + "".join(x.rstrip() + "\n\n" for x in s)
bank = fewshot.Bank(R / a.fewshot) if a.fewshot else None
def row(t, reply, idx, thinking=False):
    system = P.SYSTEM if bank or idx == 0 else P.SYSTEM_ALT
    user = t["prompt_user"]
    if bank and rng.random() < 0.5:
        shots = fewshot.block(bank.retrieve(t["question"], t["business_id"], a.fewshot_k, debt="DEBTS (" in user))
        user = user.replace("\n\nQUESTION: ", "\n\n" + shots + "\n\nQUESTION: ", 1)
    return {"system": system + "\n\nRULEBOOK:\n" + book(), "user": user, "reply": reply, "id": t["id"], "thinking": thinking}
hard, easy, think, stats = [], [], [], Counter()
for t in map(json.loads, open(R / a.plain / "trace.jsonl", encoding="utf-8")):
    g = gold[t["id"]]; good, seen = [], set()
    n = sum(1 for s in t["samples"][:8] if s["pred"] and score(s["pred"], g) == 1)
    for j, s in enumerate(t["samples"][:8]):
        if s["pred"] and score(s["pred"], g) == 1:
            reply = s["text"].split("\n#REPAIR\n")[-1].strip(); key = fewshot.skeleton(reply.split("```sql")[-1])
            if key not in seen:
                seen.add(key); good.append((reply, j // 4))
    rng.shuffle(good)
    stats[f"plain {'0' if n == 0 else '1-5' if n <= 5 else '6-8'}/8"] += 1
    if good and n <= 5:
        hard += [row(t, r, i) for r, i in good[:2]]
    elif good:
        easy.append(row(t, good[0][0], good[0][1]))
if a.think:
    for t in map(json.loads, open(R / a.think / "trace.jsonl", encoding="utf-8")):
        g = gold[t["id"]]
        ok = [(s["text"], j) for j, s in enumerate(t["samples"]) if j >= 2 and s["pred"] and score(s["pred"], g) == 1 and "\n#REPAIR\n" not in s["text"] and "</think>" in s["text"]]
        if ok:
            text, j = rng.choice(ok)
            think.append(row(t, re.sub(r"^\s*<think>\s*", "", text).strip(), (j - 2) % 2, thinking=True))
            stats["thinking question with a right sample"] += 1
        stats["thinking question"] += 1
rng.shuffle(easy); rows = think + hard + easy[:a.easy]; rng.shuffle(rows); rows = rows[:a.max]
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(dict(stats), f"| rows: hard {len(hard)}, thinking {len(think)}, easy {min(len(easy), a.easy)} -> {len(rows)} in {a.out}")
