#!/usr/bin/env bash
# Overnight RFT LoRA for the SQL generator (2026-11-02): Qwen3.5-9B samples K=8 SQL replies (no thinking) for the 2368
# questions of the 24 train shops outside holdout150, with the 0.394 prompts (+ merged vote, catalog view test); replies
# whose executed answer matches gold become SFT rows (one per question, max 1800); QLoRA r16, 1 epoch.
# Holdout150's 6 shops stay unseen for tomorrow's check. Engine: 128 seqs / 8192 batched tokens (64 seqs left 127 requests waiting).
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_q35_rft_train24
say "rft: generating K=8 samples on 2368 train questions (24 shops) -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 0 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --ids splits/rft_train24.txt --save-prompts > $OUT.log 2>&1
say "rft generation exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs; PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/rft_train24.txt)"
python3 runs/analysis/build_sft_sql.py $OUT data/train data/sft/sql_rft_v1.jsonl --max 1800 2>&1 | grep -v Warning | tee -a $LOG
say "rft: QLoRA training -> runs/lora/sql_rft_v1"
.venv-train/bin/python train/train_lora.py data/sft/sql_rft_v1.jsonl --out runs/lora/sql_rft_v1 --model sqlbase/models/Qwen3.5-9B \
  --max-len 16384 --r 16 --alpha 32 --lr 1e-4 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking > runs/lora/sql_rft_v1.log 2>&1
say "rft training exit $?: $(grep -E 'steps in|examples' runs/lora/sql_rft_v1.log | tr '\n' ' ')"
say "chain rft finished"
