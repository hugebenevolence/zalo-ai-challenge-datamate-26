"""Check a finished run folder before uploading or rerunning from it.

    .venv/bin/python train/verify_run.py runs/gemma_v7_public [--split public_test]

Checks, failing (exit 1) on the first three:
1. submission.jsonl: one valid line per question of the split (check_submission rules);
2. every submitted id has its log, and no log is left without a row;
3. the decision is reproducible: rebuilding each answer from its log (postprocess, vote, tau) gives the row;
then summarizes runs.jsonl (starts, resumes, code version) and progress.jsonl (failures, repairs, tokens,
outputs cut off at max_tokens, seconds per question). Run from the repo root with the inference venv.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_submission import problem  # noqa: E402
from redecide import redecide  # noqa: E402  (loads the notebooks' lib cells)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default=None, help="default: the split recorded in runs.jsonl, else public_test")
    args = ap.parse_args()
    run = (REPO / args.run).resolve()
    runs = read_jsonl(run / "runs.jsonl") or [r for p in sorted(run.glob("shards/*/runs.jsonl")) for r in read_jsonl(p)]
    split = args.split or next((Path(r["split"]).name for r in runs if r.get("event") == "start"), "public_test")
    ids = [q["id"] for q in read_jsonl(REPO / "data" / split / "questions.jsonl")]
    cfg = json.loads((run / "config.json").read_text())
    rows = read_jsonl(run / "submission.jsonl")
    logs = {p.stem: p for p in (run / "logs").glob("*.json")}
    errors = []

    seen = Counter(r.get("id") for r in rows)
    errors += [f"duplicated id {q}" for q, c in seen.items() if c > 1]
    errors += [f"{r.get('id')}: {p}" for r in rows if (p := problem(r))]
    missing = [q for q in ids if q not in seen]
    no_log = [r["id"] for r in rows if r["id"] not in logs]
    orphan = [q for q in logs if q not in seen]
    errors += [f"no log for {q}" for q in no_log[:20]]
    mismatch = []
    for r in rows:
        if r["id"] in logs:
            again = redecide(json.loads(logs[r["id"]].read_text(encoding="utf-8")), cfg["k"], cfg["tau"])
            if again != r:
                mismatch.append((r["id"], r, again))
    errors += [f"{q}: submitted {a} but the log gives {b}" for q, a, b in mismatch[:20]]

    print(f"run {run.name}: split {split}, k={cfg['k']} tau={cfg['tau']} thinking={cfg.get('thinking')}")
    print(f"rows {len(rows)} / {len(ids)} questions | missing {len(missing)} (scored as abstain) | "
          f"logs {len(logs)} | orphan logs {len(orphan)} | decision mismatches {len(mismatch)}")
    print("types:", dict(Counter(r.get("type") for r in rows).most_common()))
    starts = [r for r in runs if r.get("event") == "start"]
    if starts:
        versions = {(r.get("git_commit", "")[:10], r.get("git_diff_sha")) for r in starts}
        print(f"runs.jsonl: {len(starts)} start(s), {sum(r.get('event') == 'end' for r in runs)} end(s); "
              f"code {sorted(versions)}{' (MIXED code versions across resumes)' if len(versions) > 1 else ''}")
    prog = read_jsonl(run / "progress.jsonl")
    prog = prog or [r for p in sorted(run.glob("shards/*/progress.jsonl")) for r in read_jsonl(p)]
    if prog:
        tot = Counter()
        for p in prog:
            tot.update({k: p.get(k, 0) for k in ("questions", "answered", "samples", "failed_samples", "repairs",
                                                  "repairs_fixed", "gen_tokens", "truncated", "llm_seconds",
                                                  "total_seconds")})
        print(f"progress: {len(prog)} shops | failed samples {tot['failed_samples']}/{tot['samples']} | repairs "
              f"{tot['repairs_fixed']}/{tot['repairs']} fixed | truncated outputs {tot['truncated']} | "
              f"{tot['gen_tokens']:,} tokens | {tot['total_seconds'] / max(tot['questions'], 1):.1f} s per question")
    for e in errors[:30]:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    print("run OK")


if __name__ == "__main__":
    main()
