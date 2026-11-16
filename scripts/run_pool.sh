#!/usr/bin/env bash
# The submitted setup (public 0.800) outside Docker: the image's entry point docker/predict.sh with repository paths. It drafts
# every shop's layers, answers with LoRA v3 (pass 1), answers again with LoRA v5 on the same layers (pass 2), then pools the
# samples of both passes into one vote (analysis/pool_runs.py). The file to upload is OUT/submission.jsonl.
#   bash scripts/run_pool.sh [SPLIT_DIR] [OUT] [extra datamate.predict flags]   defaults: data/public_test runs/public_pool
# About 7 h on one 24 GB GPU (layers about 1.3 h, each pass about 3 h). NUM_GPUS, TIMEOUT_SECONDS (default 24 h here), PYTHON and
# MODEL_DIR override the defaults.
set -u
cd "$(dirname "$0")/.."
R=$PWD
DATA=$(realpath "${1:-data/public_test}")
OUT=$(realpath -m "${2:-runs/public_pool}")
shift $(( $# > 2 ? 2 : $# ))
mkdir -p "$OUT"
CODE_DIR=$R FLAGS_FILE=$R/scripts/flags.sh PYTHON=${PYTHON:-$R/.venv/bin/python} MODEL_DIR=${MODEL_DIR:-$R/models/Qwen3.5-9B} \
LORA_DIR=$R/adapters/sql_rft_v3 LORA2_DIR=$R/adapters/sql_v5 ASSETS_DIR=$R/assets POOL_SCRIPT=$R/analysis/pool_runs.py \
DATA_DIR=$DATA RESULT_DIR=$OUT TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-86400} bash docker/predict.sh "$@"
