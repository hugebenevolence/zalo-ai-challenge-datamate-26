"""PA of a submission against train answers: python runs/analysis/score_sub.py SUB.jsonl IDS.txt"""
import json, sys
sys.path.insert(0, "src")
from datamate.scoring import score_one
refs = {r["id"]: r for r in map(json.loads, open("data/train/answers.jsonl"))}
ids = open(sys.argv[2]).read().split()
rows = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
s = [score_one(rows.get(i, {"id": i, "type": "abstain"}), refs[i]) for i in ids]
print(f"{sum(s) / len(s):.3f} {len(s)} {s.count(1)} {s.count(-1)} {s.count(0)}")
