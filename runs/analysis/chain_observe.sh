#!/usr/bin/env bash
# Harness test: v2 pipeline + observe turn + thinking retry on holdout150, after the thinking-only run. Against 0.760 (121/7/22) for the
# same questions without them. Per-sample effect (wrong->right vs right->wrong) comes from the "observed" field of the trace.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain thinkonly finished" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_v2_observe150
say "observe test: v2 + observe turn + think retry on holdout150 -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_v2_holdout632 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views \
  --observe --think-retry --ids splits/holdout150.txt > $OUT.log 2>&1
say "observe exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout150.txt)   (without: 0.760 150 121 7 22)"
grep -aE "^\[observe\]" $OUT.log | tee -a $LOG
say "chain observe finished"
