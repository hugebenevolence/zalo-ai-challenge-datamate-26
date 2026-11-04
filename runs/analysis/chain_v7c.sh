#!/usr/bin/env bash
# v7 with the 2026-11-02 fixes (tables[key] prompt, merge-suffix hint, file-stem variables, memory cap): holdout, then public if it holds up.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH DATAMATE_GPU_UTIL=0.92 DATAMATE_MAX_SEQS=32
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "starting v7b holdout"
./run.sh train runs/gemma_v7b_hold150 -p IDS ../splits/holdout150.txt -p K 5 -p THINKING 2048 > runs/gemma_v7b_hold150.log 2>&1
.venv/bin/python train/verify_run.py runs/gemma_v7b_hold150 --split train 2>&1 | grep -v Warning | tee -a $LOG
PA=$(.venv/bin/python - <<'PY'
import glob, json, sys
sys.path.insert(0, "src")
from datamate.scoring import score_one
refs = {r["id"]: r for r in map(json.loads, open("data/train/answers.jsonl"))}
logs = [json.load(open(p)) for p in glob.glob("runs/gemma_v7b_hold150/logs/*.json")]
s = [score_one(lg["final"], refs[lg["id"]]) for lg in logs]
print(f"{sum(s) / max(len(s), 1):.3f} {len(s)} {s.count(1)} {s.count(-1)} {s.count(0)}")
PY
)
say "v7b holdout: PA N right wrong abstain = $PA   (v5 0.813 | v6 replay 0.820 | v7 0.767)"
if .venv/bin/python -c "import sys; v=sys.argv[1].split(); sys.exit(0 if len(v)==5 and int(v[1])==150 and float(v[0])>=0.80 else 1)" "$PA"; then
  say "holdout OK: starting v7b public"
  ./run.sh public_test runs/gemma_v7b_public -p K 5 -p THINKING 2048 > runs/gemma_v7b_public.log 2>&1
  .venv/bin/python train/verify_run.py runs/gemma_v7b_public 2>&1 | grep -v Warning | tee -a $LOG
else
  say "holdout below 0.80 or incomplete: v7b public NOT started"
fi
say "chain finished"
