#!/usr/bin/env bash
# After chain_morning (pid 275143): pooled votes of the base 9B and the RFT-LoRA runs (samples of both, one vote), no GPU.
# A pooled 14B+9B vote on holdout150 gave 0.693 (107 right / 3 wrong) against 0.673 / 0.680 alone.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "chain pool: waiting for chain morning"
while kill -0 275143 2>/dev/null; do sleep 60; done
if [ -f runs/sql_q35_lora1_hold150/trace.jsonl ]; then
  python3 runs/analysis/pool_vote.py runs/sql_q35_pool1_hold150 runs/sql_q35_hold150 runs/sql_q35_lora1_hold150 --split train 2>&1 | grep -v Warning | tail -1
  say "holdout150: base $(python3 runs/analysis/score_sub.py runs/sql_q35_hold150/submission.jsonl splits/holdout150.txt) | lora $(python3 runs/analysis/score_sub.py runs/sql_q35_lora1_hold150/submission.jsonl splits/holdout150.txt) | pooled $(python3 runs/analysis/score_sub.py runs/sql_q35_pool1_hold150/submission.jsonl splits/holdout150.txt)   (PA N right wrong abstain)"
fi
if [ -f runs/sql_q35_lora1_public/trace.jsonl ]; then
  python3 runs/analysis/pool_vote.py runs/sql_q35_pool1_public runs/sql_q35_public runs/sql_q35_lora1_public 2>&1 | grep -v Warning | tail -1 | tee -a $LOG
  python3 train/check_submission.py runs/sql_q35_pool1_public/submission.jsonl 2>&1 | grep -v Warning | tail -2 | tee -a $LOG
fi
say "chain pool finished: submit runs/sql_q35_pool1_public and runs/sql_q35_lora1_public"
