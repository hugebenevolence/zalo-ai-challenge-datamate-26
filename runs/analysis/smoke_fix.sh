#!/usr/bin/env bash
# GPU smoke of the sqlbase fixes (repair turns + hints, thinking-sample repair, month hints, list check): 12 dev_014
# questions with the most failed samples, views reused from runs/sql_public.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p runs/sql_smoke_fix
python3 sqlbase/predict.py --data data/public_test --output runs/sql_smoke_fix --model sqlbase/models/Qwen3-14B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
  --ids runs/analysis/smoke_fix_ids.txt --views-from runs/sql_public > runs/sql_smoke_fix.log 2>&1
echo "exit $?"
