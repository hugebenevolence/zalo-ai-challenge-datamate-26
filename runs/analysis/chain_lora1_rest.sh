#!/usr/bin/env bash
# The machine went down at 07:44 (rebooted 14:30) during runs/sql_q35_lora1_public: 11/14 shops done. Re-run the other
# 3 shops with the same settings (LoRA, fresh view for dev_014, reused views for dev_012/013) and splice them in; dev_011
# again too: its morning view passed the catalog test by writing the missed groups as NULL (the test now catches that).
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_q35_lora1_public_rest
say "lora1 public: re-running the 3 shops lost in the 07:44 shutdown -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/views_q35_fixed --lora runs/lora/sql_rft_v1/adapter --ids runs/analysis/lora1_public_rest.txt > $OUT.log 2>&1
say "rest exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[views\]|^\[dev_" $OUT.log | tee -a $LOG
python3 runs/analysis/splice.py merge runs/sql_q35_lora1_public/submission.jsonl $OUT runs/sql_q35_lora1_full 2>&1 | grep -v Warning | tee -a $LOG
cat runs/sql_q35_lora1_public/trace.jsonl $OUT/trace.jsonl > runs/sql_q35_lora1_full/trace.jsonl
python3 train/check_submission.py runs/sql_q35_lora1_full/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
say "lora1 full public ready: runs/sql_q35_lora1_full/submission.jsonl"
