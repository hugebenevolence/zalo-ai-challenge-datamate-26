#!/usr/bin/env bash
# Agent A1 on Qwen3.5-9B (the only model from 2026-11-02; Qwen3-14B dropped): --agent (UNDERSTAND + GROUND, period windows
# from the reading instead of regex hints), views of runs/sql_q35_public, on the questions that run was unsure of
# (abstained, or vote share < 0.75 on the first 8 samples); spliced into its submission, so the leaderboard delta
# measures the agent on exactly those questions. Waits for the Qwen3.5 chain; GPU smoke on dev_011's unsure questions first.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MODEL=sqlbase/models/Qwen3.5-9B
BASE=runs/sql_q35_public
run() {  # out ids
  local out=$1 ids=$2
  python3 sqlbase/predict.py --data data/public_test --output $out --model $MODEL --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
    --views-from $BASE --agent --ids $ids > $out.log 2>&1
}
say "chain agent35: waiting for the Qwen3.5 chain"
while pgrep -f "chain_q35.sh" > /dev/null || pgrep -f "output runs/sql_q35_public" > /dev/null; do sleep 60; done
if [ "$(ls $BASE/logs/q_*.json 2>/dev/null | wc -l)" != 1000 ]; then say "Qwen3.5 public run incomplete: agent NOT started"; exit 1; fi
python3 runs/analysis/splice.py uncertain $BASE > runs/analysis/uncertain_q35.txt
python3 - <<'PY'
import json
unc = set(open("runs/analysis/uncertain_q35.txt").read().split())
ids = [q["id"] for q in map(json.loads, open("data/public_test/questions.jsonl")) if q["business_id"] == "dev_011" and q["id"] in unc]
open("runs/analysis/smoke_agent35.txt", "w").write("\n".join(ids) + "\n")
PY
n=$(grep -c . runs/analysis/smoke_agent35.txt)
say "agent smoke: Qwen3.5-9B --agent on $n unsure dev_011 questions ($(wc -l < runs/analysis/uncertain_q35.txt) unsure in all)"
mkdir -p runs/sql_q35_agent_smoke && run runs/sql_q35_agent_smoke runs/analysis/smoke_agent35.txt
rc=$?
grep -aE "^\[dev_|^views of" runs/sql_q35_agent_smoke.log | tee -a $LOG
if [ $rc != 0 ] || grep -q Traceback runs/sql_q35_agent_smoke.log || [ "$(ls runs/sql_q35_agent_smoke/logs/q_*.json 2>/dev/null | wc -l)" != "$n" ]; then
  say "agent smoke failed (exit $rc): the unsure run NOT started, see runs/sql_q35_agent_smoke.log"; exit 1
fi
say "agent smoke OK: starting the unsure questions"
mkdir -p runs/sql_q35_agent_unc && run runs/sql_q35_agent_unc runs/analysis/uncertain_q35.txt
say "agent unsure run exit $?: $(ls runs/sql_q35_agent_unc/logs/q_*.json | wc -l) logs"
grep -aE "^\[dev_" runs/sql_q35_agent_unc.log | tee -a $LOG
python3 runs/analysis/splice.py merge $BASE/submission.jsonl runs/sql_q35_agent_unc runs/sql_q35_agent_spliced 2>&1 | grep -v Warning | tee -a $LOG
python3 train/check_submission.py runs/sql_q35_agent_spliced/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
say "chain agent35 finished"
