#!/usr/bin/env bash
# View fix on the two phone shops (dev_011, dev_014): their v_sales joined the abbreviation sheet on the abbreviation only,
# so lines written with the full name lost their product group ('khác' on 10% / 'không rõ' on 27% of lines). llmviews now
# rejects a draft whose product_group holds a constant no table has (repair turn names the fix) and VIEW_SYSTEM asks to
# match the catalog on code/abbreviation and full name. Base mode (no agent) with fresh views for those 2 shops only,
# spliced into the 0.394 file, so the leaderboard delta measures the view fix on their 103 questions.
# Waits for chain agent35 v3 (pid 243005).
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_q35_viewfix_phone
say "chain viewfix: waiting for chain agent35 v3"
while kill -0 243005 2>/dev/null; do sleep 60; done
say "viewfix: Qwen3.5-9B base mode, fresh views, dev_011 + dev_014 (103 questions) -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
  --ids runs/analysis/phone_shops.txt > $OUT.log 2>&1
say "viewfix exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[views\]|^\[dev_|^view drafts" $OUT.log | tee -a $LOG
python3 runs/analysis/splice.py merge runs/sql_q35_mergevote/submission.jsonl $OUT runs/sql_q35_viewfix_spliced 2>&1 | grep -v Warning | tee -a $LOG
python3 train/check_submission.py runs/sql_q35_viewfix_spliced/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
say "chain viewfix finished"
