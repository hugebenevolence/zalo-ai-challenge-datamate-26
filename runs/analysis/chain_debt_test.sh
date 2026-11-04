#!/usr/bin/env bash
# Debt views test: after the 632-question evaluation, LoRA v1 + --debt-views on the 52 debt questions of the 6 holdout shops
# (views reused from the LoRA runs, debt tables drafted fresh), against the same questions without debt views.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "debt test: waiting for the evaluation chain"
until grep -q "chain eval600 finished" $LOG; do sleep 30; done
sleep 20
OUT=runs/sql_q35_debt_test
say "debt test: LoRA v1 + debt views on 52 holdout debt questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_q35_lora1_rest450 --lora runs/lora/sql_rft_v1/adapter --debt-views --ids splits/holdout_debt.txt > $OUT.log 2>&1
say "debt test exit $?"
grep -aE "^\[debts\]|^debt drafts|^\[biz" $OUT.log | tee -a $LOG
python3 - <<'PY' 2>&1 | grep -v Warning | tee -a $LOG
import json, sys
from collections import Counter
sys.path[:0] = ["sqlbase"]
from evaluation import score
gold = {x["id"]: x for x in map(json.loads, open("data/train/answers.jsonl"))}
ids = open("splits/holdout_debt.txt").read().split()
def sub(*paths):
    out = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if r["type"] != "abstain" or r["id"] not in out:
                out[r["id"]] = r
    return out
for name, s in (("LoRA without debt views", sub("runs/sql_q35_lora1_hold150/submission.jsonl", "runs/sql_q35_lora1_rest450/submission.jsonl")), ("LoRA with debt views", sub("runs/sql_q35_debt_test/submission.jsonl"))):
    c = Counter(score(s.get(i, {"type": "abstain"}), gold[i]) for i in ids)
    print(f"debt questions, {name}: right {c[1]} wrong {c[-1]} abstain {c[0]} of {len(ids)}")
PY
say "chain debt test finished"
