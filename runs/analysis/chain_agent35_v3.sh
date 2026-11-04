#!/usr/bin/env bash
# Agent v3 on Qwen3.5-9B over ALL public questions, views of runs/sql_q35_public. v3 over v2 (runs/sql_q35_agent2_public):
# relative time phrases take relative kinds (v2 wrote 126 of them as month numbers, so 'cùng kỳ tháng trước' got the whole
# month), a month that is last month gets the rulebook pair anyway, T5/2025 = tháng 5 (v2 read 15 as quarters) with a re-ask
# when a month/quarter number is not in its text, a re-ask when a range/date has no written date, and the merged
# number/percent vote in predict.vote.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_q35_agent3_public
say "chain agent35 v3: starting (GPU idle)"

say "agent v3: Qwen3.5-9B --agent on all public questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_q35_public --agent > $OUT.log 2>&1
say "agent v3 exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[dev_|^types:|^answered|^format" $OUT.log | tee -a $LOG
say "chain agent35 v3 finished"
