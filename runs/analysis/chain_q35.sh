#!/usr/bin/env bash
# Model swap: Qwen3.5-9B (9.65B, leaves ~5.3B of the 15B budget for a second model) in the SQL pipeline with the E1 fixes,
# same settings as runs/sql_public otherwise (K=8 + 4 thinking, threshold 0.5); views drafted by the 9B itself.
# GPU smoke on dev_009 first; the full public run only starts if the smoke finishes cleanly.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MODEL=sqlbase/models/Qwen3.5-9B
run() {  # split out [extra args]
  local split=$1 out=$2; shift 2
  python3 sqlbase/predict.py --data data/$split --output $out --model $MODEL --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
    "$@" > $out.log 2>&1
}
say "chain q35: GPU smoke of Qwen3.5-9B on dev_009"
mkdir -p runs/sql_q35_smoke && run public_test runs/sql_q35_smoke --ids runs/analysis/smoke_dev009.txt
rc=$?
grep -aE "^\[dev_|^\[views\]|^model ready|^view drafts" runs/sql_q35_smoke.log | tee -a $LOG
if [ $rc != 0 ] || grep -q Traceback runs/sql_q35_smoke.log || [ "$(ls runs/sql_q35_smoke/logs/q_*.json 2>/dev/null | wc -l)" != 21 ]; then
  say "Qwen3.5 smoke failed (exit $rc): full run NOT started, see runs/sql_q35_smoke.log"; exit 1
fi
say "Qwen3.5 smoke OK: starting the full public run"
mkdir -p runs/sql_q35_public && run public_test runs/sql_q35_public
say "Qwen3.5 public exit $?: $(grep -c . runs/sql_q35_public/submission.jsonl) rows, $(ls runs/sql_q35_public/logs/q_*.json | wc -l) logs"
grep -aE "^\[dev_|^view drafts" runs/sql_q35_public.log | tee -a $LOG
python3 train/check_submission.py runs/sql_q35_public/submission.jsonl 2>&1 | grep -v Warning | tail -3 | tee -a $LOG
say "chain q35 finished"
