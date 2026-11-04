#!/usr/bin/env bash
# SQL baseline (sqlbase/, from code.zip, Qwen3-14B), public first: after the model download, run public_test with the
# predict.sh settings on one GPU, check the file; then a labeled train subset (holdout150) for per-question error
# analysis only, not as a gate. Attention runs on TRITON_ATTN: FlashInfer JIT needs nvcc >= 12.9 for sm120 and this
# machine has 12.8 (the vllm-openai:v0.32.0 image in sqlbase/Dockerfile uses FlashInfer).
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MODEL=sqlbase/models/Qwen3-14B
say "chain sql2: waiting for the Qwen3-14B download"
while pgrep -f "hf download Qwen/Qwen3-14B" > /dev/null; do sleep 30; done
want=$(python3 -c "import json; print(len(set(json.load(open('$MODEL/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 0)
have=$(ls $MODEL/*.safetensors 2>/dev/null | wc -l)
if [ "$want" = 0 ] || [ "$have" != "$want" ]; then say "Qwen3-14B download incomplete ($have/$want shards): stopping"; exit 1; fi
say "Qwen3-14B ready ($have/$want shards, $(du -sh $MODEL | cut -f1))"
run() {  # split out [extra args]
  local split=$1 out=$2; shift 2
  python3 sqlbase/predict.py --data data/$split --output $out --model $MODEL --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}' "$@" > $out.log 2>&1
}
say "starting SQL baseline on public_test"
mkdir -p runs/sql_public && run public_test runs/sql_public
say "SQL public exit $?: $(grep -c . runs/sql_public/submission.jsonl) rows, $(ls runs/sql_public/logs | wc -l) logs"
tail -3 runs/sql_public.log | cut -c1-300 | tee -a $LOG
python3 train/check_submission.py runs/sql_public/submission.jsonl 2>&1 | grep -v Warning | tail -3 | tee -a $LOG
say "starting SQL baseline on holdout150 (error analysis only)"
mkdir -p runs/sql_hold150 && run train runs/sql_hold150 --ids splits/holdout150.txt
say "SQL train subset: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py runs/sql_hold150/submission.jsonl splits/holdout150.txt)   (analysis only)"
say "chain sql2 finished"
