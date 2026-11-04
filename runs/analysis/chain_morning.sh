#!/usr/bin/env bash
# Morning of 2026-11-03, after chain_rft (pid 271345): (1) the RFT LoRA on holdout150 (6 train shops it never saw), to set
# against 0.673 for the same configuration without LoRA (runs/sql_q35_hold150); (2) the full public run with the LoRA,
# merged vote, and fresh views for the two phone shops (catalog test) while the other 12 reuse runs/sql_q35_public's.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
LORA=runs/lora/sql_rft_v1/adapter
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
say "chain morning: waiting for chain rft"
while kill -0 271345 2>/dev/null; do sleep 60; done
if [ ! -f $LORA/adapter_model.safetensors ]; then say "no adapter at $LORA: morning runs NOT started"; exit 1; fi
OUT=runs/sql_q35_lora1_hold150
say "lora check: holdout150 with $LORA -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine "$ENGINE" --ids splits/holdout150.txt --lora $LORA > $OUT.log 2>&1
say "lora holdout150 exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout150.txt)   (no LoRA: 0.673 150 107 6 37)"
python3 runs/analysis/stage_stats.py $OUT --gold data/train/answers.jsonl 2>&1 | head -1 | tee -a $LOG
OUT=runs/sql_q35_lora1_public
say "public: LoRA + merged vote + fresh phone-shop views -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/views_q35_fixed --lora $LORA > $OUT.log 2>&1
say "public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[views\]|^\[dev_|^answered|^format" $OUT.log | tee -a $LOG
python3 train/check_submission.py $OUT/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
say "chain morning finished"
