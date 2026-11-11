#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run inside the GPU allocation and matching training image. Not run locally.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a supported small dense HF checkpoint}"
: "${FILEROOT:?Set FILEROOT to a writable validation output directory}"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  tests/torchrun/run_megatron_stream_update.py \
  --model-path "$MODEL_PATH" --tp-size "${TP_SIZE:-1}" \
  --bridge-type "${BRIDGE_TYPE:-mbridge}" \
  --delay-seconds "${STREAM_TEST_TAIL_SECONDS:-30}" \
  --output-dir "$FILEROOT/minimal" "$@"
