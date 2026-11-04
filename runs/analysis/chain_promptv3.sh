#!/usr/bin/env bash
# Locked prompt test: 4 thinking samples per question with prompts.py SYSTEM_V3, LoRA v2 + few-shot + debt macros on the 632 holdout questions, after
# the observe test. Against the first 4 thinking samples of runs/sql_v2_thinkonly632 (old prompt). LoRA v2 learned the old prompt, so a tie favours v3.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain observe finished" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_v3_prompt632
say "prompt v3: 4 thinking samples on 632 holdout questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 2 --think-k 4 --think-below 1.01 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_v2_holdout632 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --prompt v3 --ids splits/holdout600.txt > $OUT.log 2>&1
say "prompt v3 exit $?"
python3 runs/analysis/thinkonly_vote.py $OUT --plain 2 2>&1 | grep -v Warning | tee -a $LOG
say "chain promptv3 finished"
