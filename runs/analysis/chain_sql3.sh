#!/usr/bin/env bash
# SQL baseline + fixes E1 (bare-month period hints, list item check, repair hints, 2 repair turns, repair of thinking
# samples), views reused from runs/sql_public so the run differs from the 0.359 run only by these changes.
# GPU smoke on dev_009 (21 questions) first; the full public run only starts if the smoke finishes cleanly.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MODEL=sqlbase/models/Qwen3-14B
run() {  # split out [extra args]
  local split=$1 out=$2; shift 2
  python3 sqlbase/predict.py --data data/$split --output $out --model $MODEL --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
    --views-from runs/sql_public "$@" > $out.log 2>&1
}
say "chain sql3 (E1 fixes): GPU smoke on dev_009"
mkdir -p runs/sql_e1_smoke && run public_test runs/sql_e1_smoke --ids runs/analysis/smoke_dev009.txt
rc=$?
grep -aE "^\[dev_|^views of" runs/sql_e1_smoke.log | tee -a $LOG
if [ $rc != 0 ] || grep -q Traceback runs/sql_e1_smoke.log || [ "$(ls runs/sql_e1_smoke/logs/q_*.json 2>/dev/null | wc -l)" != 21 ]; then
  say "E1 smoke failed (exit $rc): full run NOT started, see runs/sql_e1_smoke.log"; exit 1
fi
say "E1 smoke OK: starting the full public run"
mkdir -p runs/sql_e1_public && run public_test runs/sql_e1_public
say "E1 public exit $?: $(grep -c . runs/sql_e1_public/submission.jsonl) rows, $(ls runs/sql_e1_public/logs/q_*.json | wc -l) logs"
grep -aE "^\[dev_" runs/sql_e1_public.log | tee -a $LOG
python3 train/check_submission.py runs/sql_e1_public/submission.jsonl 2>&1 | grep -v Warning | tail -3 | tee -a $LOG
say "chain sql3 finished"
