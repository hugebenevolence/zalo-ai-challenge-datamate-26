#!/usr/bin/env bash
# Debt layer, second try: the debt_at text now gives ready recipes and order_total replaces total_due, because the samples summed
# total_due instead of calling debt_at (the LLM mapping itself matched gold). Same 52 questions, debt SELECTs reused.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "think-all exit" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_q35_debt_test2
say "debt retest: LoRA v1 + debt recipes on 52 debt questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_q35_debt_test --lora runs/lora/sql_rft_v1/adapter --debt-views --ids splits/holdout_debt.txt > $OUT.log 2>&1
say "debt retest exit $?: $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout_debt.txt)   (PA N right wrong abstain; no debt layer: 1 right 5 wrong, first try: 11 right 7 wrong)"
say "chain debt retest finished"
