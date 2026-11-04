#!/usr/bin/env bash
# Agent A1 (--agent: UNDERSTAND + GROUND, period windows from the reading instead of regex hints), Qwen3-14B with the views
# of runs/sql_public, on the 492 questions that run was unsure of; spliced into its submission (0.359), so the
# leaderboard delta measures the agent on exactly those questions. Waits for the Qwen3.5 chain; GPU smoke on dev_011 first.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MODEL=sqlbase/models/Qwen3-14B
run() {  # out ids
  local out=$1 ids=$2
  python3 sqlbase/predict.py --data data/public_test --output $out --model $MODEL --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
    --views-from runs/sql_public --agent --ids $ids > $out.log 2>&1
}
say "chain agent: waiting for the Qwen3.5 chain"
while pgrep -f "chain_q35.sh" > /dev/null || pgrep -f "output runs/sql_q35" > /dev/null; do sleep 60; done
say "agent smoke: 14B --agent on $(wc -l < runs/analysis/smoke_agent_dev011.txt) unsure dev_011 questions"
mkdir -p runs/sql_agent_smoke && run runs/sql_agent_smoke runs/analysis/smoke_agent_dev011.txt
rc=$?
grep -aE "^\[dev_|^views of" runs/sql_agent_smoke.log | tee -a $LOG
n=$(grep -c . runs/analysis/smoke_agent_dev011.txt)
if [ $rc != 0 ] || grep -q Traceback runs/sql_agent_smoke.log || [ "$(ls runs/sql_agent_smoke/logs/q_*.json 2>/dev/null | wc -l)" != "$n" ]; then
  say "agent smoke failed (exit $rc): the uncertain run NOT started, see runs/sql_agent_smoke.log"; exit 1
fi
say "agent smoke OK: starting the 492 unsure questions"
mkdir -p runs/sql_agent_unc && run runs/sql_agent_unc runs/analysis/uncertain_sql_public.txt
say "agent unsure run exit $?: $(ls runs/sql_agent_unc/logs/q_*.json | wc -l) logs"
grep -aE "^\[dev_" runs/sql_agent_unc.log | tee -a $LOG
python3 runs/analysis/splice.py merge runs/sql_public/submission.jsonl runs/sql_agent_unc runs/sql_public_agent_unc 2>&1 | grep -v Warning | tee -a $LOG
python3 train/check_submission.py runs/sql_public_agent_unc/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
say "chain agent finished"
