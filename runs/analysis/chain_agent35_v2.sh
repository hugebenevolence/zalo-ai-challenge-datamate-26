#!/usr/bin/env bash
# Agent v2 on Qwen3.5-9B over ALL public questions (the pipeline as it would be submitted), views of runs/sql_q35_public.
# v2 over v1 (runs/sql_q35_agent_unc): abbreviated time words in the UNDERSTAND prompt, a retry when an entity holds a
# time phrase, entities that are time phrases skipped in GROUND, each value says how it matched (exactly / contains it /
# similar spelling) with near spellings dropped when the phrase itself is in the data, a softer "no value" line, and the
# current month/quarter/year cut at today. Waits for chain_agent35 (pid 185562) to free the GPU.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
OUT=runs/sql_q35_agent2_public
say "chain agent35 v2: waiting for chain agent35"
while kill -0 185562 2>/dev/null; do sleep 60; done
say "agent v2: Qwen3.5-9B --agent on all public questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_q35_public --agent > $OUT.log 2>&1
say "agent v2 exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[dev_|^types:|^answered|^format" $OUT.log | tee -a $LOG
say "chain agent35 v2 finished"
