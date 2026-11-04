#!/usr/bin/env bash
# Few-shot generator test: LoRA v1, samples 4-7 (and half the thinking samples) get 4 similar train examples from the 24 shops
# instead of the raw-tables prompt, on holdout150 whose 6 shops are in neither the bank nor the LoRA. Against 0.720 (115/7/28).
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain debt test finished" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_q35_fewshot_hold150
say "few-shot test: LoRA v1 + few-shot generator on holdout150 -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_q35_lora1_hold150 --lora runs/lora/sql_rft_v1/adapter --fewshot sqlbase/fewshot_bank24.jsonl --ids splits/holdout150.txt > $OUT.log 2>&1
say "few-shot exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout150.txt)   (without: 0.720 150 115 7 28)"
python3 runs/analysis/stage_stats.py $OUT --gold data/train/answers.jsonl 2>&1 | head -3 | tee -a $LOG
say "chain fewshot test finished"
