#!/usr/bin/env bash
# Would thinking on every question catch the confident wrong answers? LoRA v1, thinking samples on all of holdout150
# (--think-below 1.01), views of the LoRA run. Analysis only. Runs after the few-shot test.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain fewshot test finished" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_q35_thinkall_hold150
say "think-all: LoRA v1, thinking on every holdout150 question -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --think-below 1.01 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_q35_lora1_hold150 --lora runs/lora/sql_rft_v1/adapter --ids splits/holdout150.txt > $OUT.log 2>&1
say "think-all exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout150.txt)   (thinking on hard questions only: 0.720 150 115 7 28)"
