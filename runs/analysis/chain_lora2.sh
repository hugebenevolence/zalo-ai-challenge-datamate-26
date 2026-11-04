#!/usr/bin/env bash
# Overnight 2026-11-03: LoRA v2 and the v2 pipeline. Thinking samples on the 1202 hard train questions of the 24 shops (few-shot and
# debt layer in the prompts, base weights), right replies + hard plain replies -> SFT rows, QLoRA continued from v1, then the v2
# pipeline on the 632 questions of the 6 holdout shops and on public. Nothing is submitted here.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
BANK=sqlbase/fewshot_bank24.jsonl
until grep -q "chain debt retest finished" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_q35_think_hard
say "lora2: thinking samples on 1202 hard train questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 2 --think-k 4 --think-below 1.01 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/sql_q35_rft_train24 --fewshot $BANK --debt-views --ids splits/rft_hard.txt --save-prompts > $OUT.log 2>&1
say "lora2 generation exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[debts\]|^debt drafts" $OUT.log | tail -30 | tee -a $LOG
python3 runs/analysis/build_sft_v2.py --plain runs/sql_q35_rft_train24 --think $OUT --fewshot $BANK --out data/sft/sql_rft_v2.jsonl --max 2400 --easy 300 2>&1 | grep -v Warning | tail -1 | tee -a $LOG
say "lora2: QLoRA continued from v1 -> runs/lora/sql_rft_v2"
.venv-train/bin/python train/train_lora.py data/sft/sql_rft_v2.jsonl --out runs/lora/sql_rft_v2 --model sqlbase/models/Qwen3.5-9B --init-adapter runs/lora/sql_rft_v1/adapter \
  --max-len 20000 --r 16 --alpha 32 --lr 5e-5 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking > runs/lora/sql_rft_v2.log 2>&1
say "lora2 training exit $?: $(grep -E 'steps in|examples,|continuing from' runs/lora/sql_rft_v2.log | tr '\n' ' ' | cut -c1-260)"
if [ ! -f runs/lora/sql_rft_v2/adapter/adapter_model.safetensors ]; then say "no v2 adapter: v2 evaluation NOT started"; exit 1; fi
LORA=runs/lora/sql_rft_v2/adapter
mkdir -p runs/views_hold_v2/logs; cp runs/sql_q35_lora1_rest450/logs/_views_biz_*.json runs/views_hold_v2/logs/; cp runs/sql_q35_debt_test/logs/_debt_biz_*.json runs/views_hold_v2/logs/
OUT=runs/sql_v2_holdout632
say "v2: LoRA v2 + few-shot + debt layer on the 632 holdout questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/views_hold_v2 --lora $LORA --lora-thinking --fewshot $BANK --debt-views --ids splits/holdout600.txt > $OUT.log 2>&1
say "v2 holdout exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout600.txt)   (LoRA v1: 0.669 632 465 42 125, base: 0.598 632 423 45 164)"
OUT=runs/sql_v2_public
say "v2: public -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/views_q35_fixed --lora $LORA --lora-thinking --fewshot $BANK --debt-views > $OUT.log 2>&1
say "v2 public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
python3 train/check_submission.py $OUT/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
say "chain lora2 finished"
