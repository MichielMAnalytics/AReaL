#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Preparation and real launcher invocation; no cluster placement is guessed.
set -euo pipefail
mode=${1:?Usage: run_e2e.sh baseline|stream [launcher config overrides...]}
shift
case "$mode" in baseline|stream) ;; *) echo 'Expected baseline or stream' >&2; exit 2 ;; esac
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${FILEROOT:?Set a shared writable FILEROOT}"
: "${BASE_CONFIG:?Set BASE_CONFIG to a compatible complete training YAML}"
: "${WORKER_IMAGE:?Set WORKER_IMAGE to the tested training/inference image}"
: "${WORKER_MOUNTS:?Set WORKER_MOUNTS explicitly for your environment}"
: "${LAUNCHER:?Set LAUNCHER=local or slurm}"
case "$LAUNCHER" in local|slurm) ;; *) echo 'Expected LAUNCHER=local or slurm' >&2; exit 2 ;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export STREAM_TEST_MODE="$mode"
export STREAM_TEST_TAIL_SECONDS=${STREAM_TEST_TAIL_SECONDS:-60}
export STREAM_TEST_OUTPUT="$FILEROOT/stream-trainer-validation/$mode"
mkdir -p "$STREAM_TEST_OUTPUT"
python3 - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ['BASE_CONFIG'])
mode = os.environ['STREAM_TEST_MODE']
# These restrictions apply identically to both runs. Save the resulting YAML
# for review: Pixel's original batch advantage normalization is not equivalent.
overrides = {
    'trial_name': 'stream-validation-' + mode,
    'total_train_steps': 1,
    'total_train_epochs': 1,
    'cluster.fileroot': os.environ['FILEROOT'],
    'actor.path': os.environ['MODEL_PATH'],
    'actor.disable_dropout': True,
    'actor.dtype': 'bfloat16',
    'actor.ppo_n_minibatches': 1,
    'actor.loss_aggregation': 'token_mean',
    'actor.reward_norm': None,
    'actor.adv_norm': None,
    'actor.kl_ctl': 0.0,
    'actor.weight_update_mode': 'awex',
    'actor.megatron.ddp.overlap_grad_reduce': False,
    'actor.megatron.ddp.overlap_param_gather': False,
    'rollout.max_head_offpolicyness': 0,
    'stream_trainer.enabled': mode == 'stream',
    'critic': None,
    'ref': None,
    'teacher': None,
    'mopd': None,
    'dynamic_bs': False,
    'recover.mode': 'disabled',
    'stats_logger.wandb.mode': 'disabled',
}
for key, value in overrides.items():
    OmegaConf.update(cfg, key, value, merge=False, force_add=True)
for spec in cfg.actor.scheduling_spec:
    spec.image = os.environ['WORKER_IMAGE']
    spec.mount = os.environ['WORKER_MOUNTS']
# Each rollout replica must share its TP width and physical GPUs with actor.
# Preserve explicit user topology and rollout scheduling specs; never infer it.
if cfg.rollout.get('scheduling_spec') is not None:
    for spec in cfg.rollout.scheduling_spec:
        spec.image = os.environ['WORKER_IMAGE']
        spec.mount = os.environ['WORKER_MOUNTS']
OmegaConf.save(cfg, Path(os.environ['STREAM_TEST_OUTPUT']) / 'config.yaml')
PY
entrypoint=${ENTRYPOINT:-examples/stream_trainer/train_e2e.py}
python3 -m "areal.infra.launcher.$LAUNCHER" "$entrypoint" \
  --config "$STREAM_TEST_OUTPUT/config.yaml" "$@" \
  2>&1 | tee "$STREAM_TEST_OUTPUT/launcher.log"
