#!/usr/bin/env bash
# SQL baseline (sqlbase/, from code.zip, Qwen3-14B): after the v7b holdout and the model download, run it on
# holdout150 (scored against train answers), then on public_test. Same settings as sqlbase/predict.sh, one GPU.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MODEL=sqlbase/models/Qwen3-14B
say "chain sql: waiting for the v7b holdout and the Qwen3-14B download"
while pgrep -f "papermill notebooks/02_agent" > /dev/null || pgrep -f "hf download Qwen/Qwen3-14B" > /dev/null; do sleep 60; done
.venv/bin/python train/verify_run.py runs/gemma_v7b_hold150 --split train 2>&1 | grep -v Warning | tee -a $LOG
say "v7b holdout: PA N right wrong abstain = $(.venv/bin/python runs/analysis/score_sub.py runs/gemma_v7b_hold150/submission.jsonl splits/holdout150.txt)   (v7 0.767 | v6 replay 0.820)"
want=$(python3 -c "import json; print(len(set(json.load(open('$MODEL/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 0)
have=$(ls $MODEL/*.safetensors 2>/dev/null | wc -l)
if [ "$want" = 0 ] || [ "$have" != "$want" ]; then say "Qwen3-14B download incomplete ($have/$want shards): stopping"; exit 1; fi
run() {  # split out [extra args]
  local split=$1 out=$2; shift 2
  python3 sqlbase/predict.py --data data/$split --output $out --model $MODEL --quant fp8 --k 8 --think-k 4 \
    --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":64,"max_num_batched_tokens":4096}' "$@" > $out.log 2>&1
}
say "starting SQL baseline on holdout150"
mkdir -p runs/sql_hold150 && run train runs/sql_hold150 --ids splits/holdout150.txt
tail -1 runs/sql_hold150.log | cut -c1-300 | tee -a $LOG
say "SQL holdout: PA N right wrong abstain = $(.venv/bin/python runs/analysis/score_sub.py runs/sql_hold150/submission.jsonl splits/holdout150.txt)   (gemma v5 0.813 | v6 replay 0.820)"
say "starting SQL baseline on public_test"
mkdir -p runs/sql_public && run public_test runs/sql_public
tail -1 runs/sql_public.log | cut -c1-300 | tee -a $LOG
.venv/bin/python train/check_submission.py runs/sql_public/submission.jsonl 2>&1 | grep -v Warning | tail -3 | tee -a $LOG
say "chain sql finished"
