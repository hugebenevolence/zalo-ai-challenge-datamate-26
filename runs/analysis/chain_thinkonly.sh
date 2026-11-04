#!/usr/bin/env bash
# Architecture input (2026-11-04): LoRA v2 + few-shot + debt macros with 8 thinking samples on all 632 holdout questions (2 plain samples,
# which the offline re-vote drops), so thinking-only votes at K = 2, 4, 6, 8 can be compared with 0.723 / 0.758. Measurement only.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_v2_thinkonly632
say "think-only: LoRA v2, 8 thinking samples on 632 holdout questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 2 --think-k 8 --think-below 1.01 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_v2_holdout632 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --ids splits/holdout600.txt > $OUT.log 2>&1
say "think-only exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs; $(python3 -c "import json;print(json.load(open('$OUT/run.json'))['seconds'])" 2>/dev/null) s"
say "chain thinkonly finished"
