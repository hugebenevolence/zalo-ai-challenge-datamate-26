#!/usr/bin/env bash
# Training data for LoRA v3: LoRA v2 + prompt v3 on all 3000 train questions, 2 plain + 4 thinking samples each, with prompts saved so
# each reply is paired with the prompt that made it. Views and debt layers are reused: 24 shops from the think_hard run, 6 from holdout632.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
V=runs/views_train30/logs
mkdir -p $V
cp runs/sql_q35_think_hard/logs/_views_*.json runs/sql_q35_think_hard/logs/_debt_*.json $V/
cp runs/sql_v2_holdout632/logs/_views_*.json runs/sql_v2_holdout632/logs/_debt_*.json $V/
OUT=runs/sql_v3_gen30
say "gen30: LoRA v2 + prompt v3, 2 plain + 4 thinking on 3000 train questions, $(ls $V/_views_*.json | wc -l) shops -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 2 --think-k 4 --think-below 1.01 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/views_train30 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --prompt v3 --save-prompts > $OUT.log 2>&1
say "gen30 exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
say "chain gen30 finished"
