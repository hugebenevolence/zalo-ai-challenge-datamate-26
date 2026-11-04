"""Check a submission.jsonl against a split's questions before uploading: one line per known id, a valid
type, an answer in the format that type requires (rulebook §9), and a tally of what will be scored.

    .venv/bin/python train/check_submission.py runs/gemma_v5_public/submission.jsonl [--split public_test]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def problem(row: dict) -> str | None:
    t, a = row.get("type"), row.get("answer")
    if t not in TYPES:
        return f"unknown type {t!r}"
    if t in ("number", "percent") and not is_num(a):
        return f"{t} answer must be a JSON number, got {a!r}"
    if t == "string" and not (isinstance(a, str) and a.strip()):
        return "string answer must be a non-empty string"
    if t == "date" and not (isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a)):
        return f"date must be YYYY-MM-DD or YYYY-MM, got {a!r}"
    if t == "boolean" and not isinstance(a, bool):
        return "boolean answer must be true/false"
    if t == "list" and not (isinstance(a, list) and a and all(is_num(x) or isinstance(x, str) for x in a)):
        return "list answer must be a non-empty array of names or numbers"
    if t == "table":
        if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
            return "table answer must be {columns, rows}"
        if any(not isinstance(r, list) or len(r) != len(a["columns"]) for r in a["rows"]):
            return "every table row needs one cell per column"
    if t == "unanswerable" and "reason" in row and not isinstance(row["reason"], str):
        return "reason must be a string (rulebook §7)"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--split", default="public_test")
    args = ap.parse_args()
    ids = [json.loads(line)["id"] for line in (REPO / "data" / args.split / "questions.jsonl").open(encoding="utf-8")]
    known, seen, errors, types = set(ids), Counter(), [], Counter()
    for n, line in enumerate(args.submission.open(encoding="utf-8"), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not JSON ({e})")
            continue
        qid = row.get("id")
        seen[qid] += 1
        if qid not in known:
            errors.append(f"line {n}: unknown id {qid!r}")
        if (p := problem(row)):
            errors.append(f"line {n} ({qid}): {p}")
        types[row.get("type")] += 1
    dups = [q for q, c in seen.items() if c > 1]
    missing = [q for q in ids if q not in seen]
    print(f"{sum(seen.values())} lines, {len(seen)} ids | missing {len(missing)} (scored as abstain)"
          f" | duplicated {len(dups)}")
    print("types:", dict(types.most_common()))
    answered = sum(c for t, c in types.items() if t != "abstain")
    print(f"answered {answered} / {len(ids)} questions")
    for e in errors[:20]:
        print("ERROR", e)
    if errors or dups:
        sys.exit(1)
    print("format OK")


if __name__ == "__main__":
    main()
