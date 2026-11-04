#!/usr/bin/env bash
# Pipeline review (2026-11-02 night): the 0.394 configuration (Qwen3.5-9B, base mode, merged vote) on the 150 labeled
# train holdout questions, for per-stage error attribution only (not a gate). Code as of now: merged vote, catalog view test.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_q35_hold150
say "review: Qwen3.5-9B base mode on train holdout150 -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
  --ids splits/holdout150.txt > $OUT.log 2>&1
say "review run exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs; PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout150.txt)"
