# Same-batch Stream Trainer validation

This implementation keeps a fixed rollout batch and policy version. A complete accepted
GRPO group can run proximal-logprob recomputation, advantage preparation and backward on
a drained colocated replica. Other replicas continue generating with the unchanged
weights. All actors then synchronize gradients and perform one optimizer update; AWEX
publishes the resulting policy after the batch finishes. This is Stream Trainer style
execution, not Laminar's cross-batch asynchronous consumption or mixed policy versions.

**GPU validation has not been run locally.** The scripts below are acceptance
procedures, not evidence that numerical parity or speedup has already passed.

## Supported boundary

The opt-in `stream_trainer.enabled` defaults to false. This initial path requires:

- Dense Megatron BF16 actors with dropout disabled, PP/CP/EP equal to 1, and DP > 1.
- SGLang with the same TP and DP as actor; verified physical GPU colocation, v1
  controllers and AWEX weight updates.
- One PPO optimizer minibatch and the `token_mean` objective; KL coefficient zero.
- No batch-level reward or advantage normalization. Complete-group normalization is
  supported. No critic, reference model, teacher or MOPD.
- No DP gradient/parameter communication overlap, R3, training MTP, tree training, LoRA,
  custom advantage stages, backend routing or dynamic batch replacement.
- Strict `rollout.max_head_offpolicyness=0`.

The Pixel experiment previously discussed used batch advantage normalization. Turning
that normalization off changes the training objective. Baseline and stream must
therefore both use the same explicitly adapted configuration. A comparison against the
original batch-normalized Pixel run cannot establish unchanged training semantics. The
E2E wrapper saves both resolved override files for review and deliberately disables
reward and advantage normalization in **both** modes. Other unsupported options are
rejected; the wrapper does not silently repair arbitrary configs or choose cluster
resources.

## Real Megatron numerical comparison: at least two GPUs

Enter an allocated GPU environment with the same Megatron/AReaL/AWEX dependencies as
training. Supply a supported small dense HF checkpoint (for example a local Qwen3-0.6B
checkpoint). The output path must be writable.

```bash
export MODEL_PATH='<checkpoint visible inside the container>'
export FILEROOT='<validation output directory>'
bash examples/stream_trainer/run_minimal_gpu.sh
```

The default is DP2 × TP1. For DP2 × TP2, explicitly set `NPROC_PER_NODE=4` and
`TP_SIZE=2`. `BRIDGE_TYPE` selects the installed model bridge. The script runs the
actual `MegatronPPOActor`, PPO objective and Megatron forward/backward schedule:

1. Warm up one real update so the Adam moments are nonempty.
1. Save model weights, optimizer metadata and scheduler state on every rank. Gather FP32
   master parameters and Adam moments to DP zero for each TP shard using MCore's public
   `get_parameter_state_dp_zero()` API; complete optimizer tensors remain on those
   roots.
1. Run one ordinary synchronous update and retain its post-reduction, pre-clipping owned
   gradient shards and resulting state. The BF16 probe uses public `prepare_grads()` to
   expose valid post-reduce-scatter optimizer gradients before clipping; it then runs
   the ordinary engine optimizer step. There is no dynamic gradient scaler, so the
   repeated preparation only repeats the same copy/view.
1. Restore optimizer metadata, then scatter master parameters and Adam moments with
   public `load_parameter_state_from_dp_zero()`. All DP ranks participate in both state
   collectives, including ranks that retain no full optimizer tensors. Check the
   identical initial state, then delay the last DP replica and accumulate complete
   groups separately, with unequal chunks/token counts.
1. Finish once all replicas arrive and compare local owned gradient shards, updated
   model weights, optimizer metadata and scheduler state on every rank. DP roots also
   compare all FP32 master parameters and Adam moments against the synchronous result.

It also requires a nonzero reference gradient and verifies that an early replica
finishes backward before the artificial delay expires. Increase
`STREAM_TEST_TAIL_SECONDS` if model startup/backward itself exceeds the default
30-second window; do not remove the ordering assertion. At completion it writes one
`FILEROOT/minimal/rank-*.json` per global rank with an explicit `passed` or `failed`
status. A strict failure on any rank marks every final result failed and exits nonzero.
It also exercises the real CPU mirror process group through `stream_topology()`,
checking that each TP replica returns every rank's host and nonempty GPU UUID.

The default numerical tolerances are `rtol=0.02, atol=2e-5`, accounting for BF16
accumulation-order differences. They are configurable arguments to the shell script and
must be assessed against the chosen model's baseline numerical variance. This comparison
does not establish bitwise equivalence or RL quality.

The fixture uses a 48-token microbatch budget for sequences of length 32, 40 and 48.
Each real microbatch therefore contains one sequence in both paths, keeping its GEMM/GDN
computation shape fixed while changing chunk counts and token masses. Repacking multiple
sequences into different BF16 shapes can itself exceed the elementwise tolerance and
requires a separate numerical-variance assessment. Passing this fixture does not
validate arbitrary production repacking differences.

The default fixture retains its original 8-token prompt masks and elementwise
`rtol=0.02, atol=2e-5` checks. Earlier failed runs remain failed. Before the final
assertion it writes `diagnostics-rank-*.json`, first with gradient diagnostics and then
with updated-model, optimizer-metadata, FP32 master/moment and scheduler comparisons. A
gradient mismatch does not skip the subsequent state comparisons. Each tensor and its
rank-local aggregate report relative L2 error, cosine similarity, observed/reference
norm ratio, the least-squares scale `a = dot(reference, observed) / ||reference||²`,
residual `||observed - a*reference|| / ||observed||`, strict mismatch rate and maximum
absolute difference. The original `torch.testing.assert_close` decision is recorded
separately from those diagnostics; small aggregate errors never override a failed strict
check.

The `aggregate.sums` fields contain FP64 sufficient statistics: squared reference and
observed norms, dot product, squared difference, element/mismatch counts and nonfinite
counts. Sum the gradient fields across ranks before deriving global metrics; do not
average rank-local relative errors or cosines. TP-replicated optimizer parameters remain
counted once per rank in this diagnostic scope. Full optimizer parameter/moment states
are compared only on their DP roots, while model states are compared on every rank.
Nonfinite metric values are written as JSON `null` and cannot qualify a failed
comparison as passed. Diagnostics have `status: collecting` until all comparisons and
the global failure decision complete, so incomplete output is not a passing result.

A separate **DP2-only normalization-rounding control** keeps the 32/40/48 sequence
lengths and 48-token microbatch budget, but changes group 0/1/2 prompt-mask prefixes to
16/16/24. This gives global valid-token mass 128 and local DP masses 80/48, rather than
the default global mass 192. Invoke it with a distinct output directory:

```bash
FILEROOT='<separate normalization-control output directory>' \
  bash examples/stream_trainer/run_minimal_gpu.sh --power-of-two-mass
```

The script labels this output `fixed-shape-power-of-two-mass` and records actual local
and global token masses, checking the expected 80/48 and 128 explicitly. Default output
is labeled `fixed-shape-default-mass`. The control makes the legacy backward loss scale
`DP/global_mass = 1/64`, a binary power, to isolate BF16 rounding when normalization
moves from before backward to after gradient accumulation. It does not loosen
thresholds, enable a different model/DDP normalization path, or change the production
engine. **A passing control does not establish that the default legacy fixture or
arbitrary production packing/normalization passes numerical parity.**

An additional optional **independent serial numerator reference B** separates legacy
BF16 loss scaling from streamed scheduling:

```bash
FILEROOT='<separate independent-reference output directory>' \
  bash examples/stream_trainer/run_minimal_gpu.sh --serial-numerator-reference
```

Without this option, reference **A** remains the ordinary synchronous PPO update and is
labeled `A_original_serial`. With the option, warmup and the initial snapshot stay
unchanged, but the compared serial update uses a test subclass's two hooks: the normal
`train_batch()` still constructs microbatches and computes the true global denominator
`M`; its loss callback substitutes denominator 1 and divides its original `DP*Nmb`
multiplier by DP. MCore's normal `/Nmb` then leaves a raw token numerator backward.
Immediately before the ordinary serial gradient finalization, the test scales FP32 main
gradients by `DP/M`; the normal DP average and real optimizer step remain intact. This
reference invokes neither `stream_accumulate()` nor `stream_backward_batch()` to
construct its gradients. It only supports the legacy `calculate_per_token_loss=False`
path and does not change that model/DDP setting.

B is labeled `B_serial_numerator`, adds `-serial-numerator-reference` to the fixture
label and sets `original_A_pass_claimed: false`. Its actual global mass, callback count,
FP32 scale and single-finalization count are recorded. It can be combined with the
explicit power-of-two control, still using a separate output directory. **Passing
B-versus-stream does not change a failed A-versus-stream result into a pass and does not
establish parity with the original legacy production normalization.**

Every run now records the ordered microbatches passed to the real forward/backward
schedule and actual loss callbacks. SHA-256 fingerprints include tensor dtype, shape and
bytes for input IDs, sequence boundaries, masks, advantages and behavior/proximal
logprobs when present. Both original and source padded model inputs are recorded,
alongside padding metadata, native-normalization flags and per-sequence semantic hashes
that exclude unrelated padded rows. Transport dummy microbatches remain visible but are
excluded from real-sequence order comparisons. These are inputs at the engine's schedule
boundary; they do not claim to fingerprint intermediates inside model kernels. For B, a
pass additionally requires matching real forward/backward padded inputs and semantic
sequence order, including the loss inputs. Mismatching hashes are reported; they must
not be described as a pure normalization-placement experiment.

DP roots also report `master_update_delta`: chunked comparisons of
`serial_master - initial_master` against `stream_master - initial_master`, including
aggregate relative L2 and cosine. This avoids a small update difference being obscured
by the absolute parameter magnitude. It is labeled `diagnostic_only`, uses the same
stated tolerances for its descriptive mismatch fields, and does not alter the strict
acceptance decision. Delta tensors are computed one bounded chunk at a time, without
keeping additional complete model snapshots resident.

This minimal test uses Megatron's real **distributed Adam** optimizer. Its ordinary
`state_dict()` contains metadata but omits the sharded master parameters and moments;
the separate public DP-zero APIs are required for a complete snapshot and restore. The
test requires nonzero `exp_avg` and `exp_avg_sq`, FP32 optimizer tensors, a nonzero
reference gradient and an actual change to FP32 master parameters. It checks the
restored snapshot exactly before comparing updates with the stated BF16 tolerances.
Non-owned regions of a model's full `main_grad` buffer are deliberately excluded after
reduce-scatter because they are not valid global gradients.

It instantiates the real AWEX adapter/residency without publishing metadata to an
inference service. It does not validate AWEX transfers or offload races. The E2E path
below covers actual SGLang/AWEX orchestration, but its stochastic rollout outputs do not
constitute an optimizer-state parity test. No optimizer method or production checkpoint
assertion is patched or bypassed.

## Real colocated SGLang/AWEX E2E

Use a complete existing config whose topology meets the supported boundary. Actor
DP2/TP1 and rollout DP2/TP1 is the smallest topology. The older
`examples/math/gsm8k_grpo_awex_colocate.yaml` is not directly compatible: its rollout
DP1/TP2 differs from actor DP2/TP1. Set both explicitly in a dedicated config; do not
infer topology from the number of visible devices.

```bash
export MODEL_PATH='<checkpoint visible to actor and SGLang>'
export BASE_CONFIG='<complete compatible training YAML>'
export FILEROOT='<shared experiment filesystem>'
export WORKER_IMAGE='<tested GPU worker image>'
export WORKER_MOUNTS='<explicit worker mounts>'
export LAUNCHER=local  # or slurm with placement supplied in BASE_CONFIG/CLI
export STREAM_TEST_TAIL_SECONDS=60
bash examples/stream_trainer/run_e2e.sh baseline
bash examples/stream_trainer/run_e2e.sh stream
```

Run `local` inside an existing GPU allocation/container. The local launcher uses the
current runtime; `WORKER_IMAGE` and `WORKER_MOUNTS` configure scheduler specs for
launchers that create containers. For Slurm, supply reservation, partition, node
constraints and all required mounts through your actual config/launcher arguments. This
script does not allocate a reservation, modify drivers or guess cluster placement. The
checkout must be visible to workers at the configured path/PYTHONPATH.

The example constructs one fixed batch with at least two prompt groups. All samples in
its last group wait asynchronously before issuing a real SGLang model request. Fast
groups issue real requests and collect actual token logprobs through `RLVRWorkflow`. The
reward is constant and normalization is disabled so the smoke test has a nonzero
training signal. The delayed workflow is only a test fixture; it is not used by normal
Pixel training. Both runs perform one update. Their YAMLs and launcher logs are saved
below `FILEROOT/stream-trainer-validation/{baseline,stream}`. Worker logs remain under
the normal experiment/trial log directory configured by AReaL.

Check the **trainer log** containing coordinator markers:

```bash
python3 examples/stream_trainer/check_timeline.py '<stream trainer main.log>'
```

For each update the checker requires backward completion before `rollout_complete`, all
backwards and rollout completion before `optimizer_start`, and exactly one
`optimizer_start`/`optimizer_done`. Markers use a single coordinator monotonic time axis
(`elapsed_s`). Also inspect worker/server logs to verify the slow SGLang replica remains
live during early actor backward and that one AWEX update publishes only after optimizer
completion. Coordinator markers alone cannot prove GPU overlap or weight-version
correctness; use server request versions and a profiler for that acceptance step. A
failed marker check, OOM, deadlock or unsupported config is a failed E2E validation, not
a performance result.

To exercise the real Pixel workflow and dataset instead of the artificial single-turn
fixture, provide its compatible full config and entrypoint:

```bash
export BASE_CONFIG='<adapted Pixel config>'
export ENTRYPOINT=examples/vlm/pixel_reasoner.py
bash examples/stream_trainer/run_e2e.sh baseline
bash examples/stream_trainer/run_e2e.sh stream
```

Keep dataset order, seed, model, generation limits, accepted group count and
normalization identical. Pixel uses its real natural tail; the fixture's artificial
delay is not injected into Pixel. Verify that no samples were dropped or replaced, all
sampled tokens carry the original policy version, and the final policy version advances
once. The numerical script compares one fixed synthetic tensor batch; separate E2E
stochastic rollouts must not be advertised as the same tensor batch. Report complete
step wall time including offload/onload, logprob recomputation, backward, final
synchronization and AWEX, alongside any overlap counters. A long tail does not guarantee
speedup when it also owns most of the training token mass.
