#!/usr/bin/env bash
# The 0.491 public setup end to end: LoRA v2, the few-shot bank, the views and debt layers saved from that run, then debt routing.
# Usage: bash runs/analysis/run_v2_public.sh [OUT]. The file to submit is OUT_debtroute/submission.jsonl. Set PYTHON if python3 is not the project's venv.
set -eu
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PYTHON:-python3}
OUT=${1:-runs/v2_public}
mkdir -p "$OUT"
"$PY" sqlbase/predict.py --data data/public_test --output "$OUT" --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from sqlbase/config_v2_public --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views
"$PY" runs/analysis/debt_vote.py "$OUT" --write "${OUT}_debtroute"
"$PY" train/check_submission.py "${OUT}_debtroute/submission.jsonl"
