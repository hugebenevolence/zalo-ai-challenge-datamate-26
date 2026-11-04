#!/usr/bin/env bash
# After the v6 public run: v7 on the holdout; v7 on public only if the holdout PA did not drop.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH DATAMATE_GPU_UTIL=0.92 DATAMATE_MAX_SEQS=32
LOG=runs/analysis/chain_v7.log
echo "$(date) waiting for the v6 public run" | tee -a $LOG
until grep -q "format OK\|ERROR" runs/gemma_v6_public.log 2>/dev/null; do sleep 60; done
echo "$(date) v6 public done; starting v7 holdout" | tee -a $LOG
./run.sh train runs/gemma_v7_hold150 -p IDS ../splits/holdout150.txt -p K 5 -p THINKING 2048 > runs/gemma_v7_hold150.log 2>&1
PA=$(.venv/bin/python - <<'PY'
import glob, json, sys
sys.path.insert(0, "src")
from datamate.scoring import score_one
refs = {r["id"]: r for r in map(json.loads, open("data/train/answers.jsonl"))}
logs = [json.load(open(p)) for p in glob.glob("runs/gemma_v7_hold150/logs/*.json")]
s = [score_one(lg["final"], refs[lg["id"]]) for lg in logs]
print(f"{sum(s) / max(len(s), 1):.3f} {len(s)} {s.count(1)} {s.count(-1)} {s.count(0)}")
PY
)
echo "$(date) v7 holdout: PA N right wrong abstain = $PA" | tee -a $LOG
if .venv/bin/python -c "import sys; v=sys.argv[1].split(); sys.exit(0 if len(v)==5 and int(v[1])==150 and float(v[0])>=0.80 else 1)" "$PA"; then
  echo "$(date) holdout OK; starting v7 public" | tee -a $LOG
  ./run.sh public_test runs/gemma_v7_public -p K 5 -p THINKING 2048 > runs/gemma_v7_public.log 2>&1
  .venv/bin/python train/check_submission.py runs/gemma_v7_public/submission.jsonl 2>&1 | tee -a $LOG
else
  echo "$(date) holdout below 0.80 or incomplete: v7 public NOT started" | tee -a $LOG
fi
echo "$(date) chain finished" | tee -a $LOG
