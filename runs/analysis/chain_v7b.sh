#!/usr/bin/env bash
# After the v6 public run: GPU smoke of the new code, v7 holdout, then v7 public only if the holdout held up.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH DATAMATE_GPU_UTIL=0.92 DATAMATE_MAX_SEQS=32
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "chain v7b: waiting for the v6 public run"
until grep -q "format OK\|ERROR\|Traceback" runs/gemma_v6_public.log 2>/dev/null; do sleep 60; done
.venv/bin/python train/verify_run.py runs/gemma_v6_public 2>&1 | grep -v Warning | tee -a $LOG
say "GPU smoke of the v7 code (6 questions, K=2)"
rm -rf runs/smoke_v7_gpu
./run.sh train runs/smoke_v7_gpu -p IDS ../splits/smoke6.txt -p K 2 -p THINKING 2048 > runs/smoke_v7_gpu.log 2>&1
if ! .venv/bin/python train/verify_run.py runs/smoke_v7_gpu --split train 2>&1 | grep -v Warning | tee -a $LOG | grep -q "run OK"; then
  say "GPU smoke FAILED: stopping before the holdout (see runs/smoke_v7_gpu/shard0.txt)"; exit 1
fi
say "starting v7 holdout"
./run.sh train runs/gemma_v7_hold150 -p IDS ../splits/holdout150.txt -p K 5 -p THINKING 2048 > runs/gemma_v7_hold150.log 2>&1
.venv/bin/python train/verify_run.py runs/gemma_v7_hold150 --split train 2>&1 | grep -v Warning | tee -a $LOG
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
say "v7 holdout: PA N right wrong abstain = $PA   (v5 run 0.813 | v6 replay 0.820)"
if .venv/bin/python -c "import sys; v=sys.argv[1].split(); sys.exit(0 if len(v)==5 and int(v[1])==150 and float(v[0])>=0.80 else 1)" "$PA"; then
  say "holdout OK: starting v7 public"
  ./run.sh public_test runs/gemma_v7_public -p K 5 -p THINKING 2048 > runs/gemma_v7_public.log 2>&1
  .venv/bin/python train/verify_run.py runs/gemma_v7_public 2>&1 | grep -v Warning | tee -a $LOG
else
  say "holdout below 0.80 or incomplete: v7 public NOT started"
fi
say "chain finished"
