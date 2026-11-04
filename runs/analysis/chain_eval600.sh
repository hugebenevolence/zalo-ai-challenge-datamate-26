#!/usr/bin/env bash
# Model evaluation (2026-11-03): LoRA v1 and the base 9B on the 450 questions of the 6 holdout shops not in holdout150,
# so both have all 600 questions of shops the LoRA never saw (the 150 already exist: runs/sql_q35_lora1_hold150, sql_q35_hold150).
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
for cfg in "lora1 --lora runs/lora/sql_rft_v1/adapter" "base"; do
  set -- $cfg; name=$1; shift
  OUT=runs/sql_q35_${name}_rest450
  say "eval600: $name on holdout_rest450 -> $OUT"
  mkdir -p $OUT
  python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine "$ENGINE" --ids splits/holdout_rest450.txt "$@" > $OUT.log 2>&1
  say "eval600 $name exit $?: rest450 PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout_rest450.txt)"
done
say "chain eval600 finished"
