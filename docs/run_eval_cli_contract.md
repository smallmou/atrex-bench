# `run_eval` CLI and Configuration Contract

## 1. Entry Points

CLI integrations only need to expose one option:

```bash
python scripts/run_eval.py --config /abs/path/run_eval.json
```

Existing public CLI options remain supported. Explicit CLI options take precedence over the config. Absolute paths are recommended for all path fields.

For both CLI and SDK calls, relative `input`, `reference_dir`, and `output` paths
are resolved against the caller's working directory before starting workers
(not against the config file or package directory). A relative `checkpoint_dir`
is instead based on the current run's artifact directory:
`<output>/<timestamp>/<kernel>/`. If omitted, that artifact directory is also the
checkpoint root. Absolute checkpoint paths are used as supplied.

Python callers can use `atrex_bench.evaluate(config)`. The SDK accepts the same fields as
`--config`, runs the evaluator in a separate subprocess, and returns the complete
`eval_result.json` object. Normal compile, correctness, or performance stage failures
are returned as results. SDK exceptions are raised only when the configuration cannot
start an evaluation, the subprocess fails abnormally, or the result artifacts are invalid.

## 2. Configuration Fields

| Field | Type | Required when | Description |
|---|---|---|---|
| `schema_version` | string | Recommended | Currently fixed to `v1`. |
| `eval_mode` | enum | Optional | `candidate` or `torch_compile_reference`; defaults to `candidate`. |
| `validation_mode` | enum | Optional | `full`, `correctness_only`, or `performance_only`; defaults to `full`. |
| `input` | path | Required in candidate mode | Candidate Python file; must not be set in Torch compile mode. |
| `reference_dir` | path | Always | Reference directory. Must contain `reference.py`, `input.py`, `shapes.json`, and `metadata.json`. |
| `output` | path | Always | Root directory for evaluation artifacts. |
| `checkpoint_dir` | path | Optional | Root directory for correctness/performance checkpoints. |

The candidate file must expose `class Model`.

Except for mode switches and compatibility aliases, all public CLI options below can also be set in the config. Use snake_case field names without the leading `--`. For example, `--warmup-iters` maps to `warmup_iters`.

## 3. Evaluation Modes

| Mode | CLI options | Stages |
|---|---|---|
| Full | No only-mode option | Compile, correctness, and performance. |
| Correctness only | `--correctness-only` | Compile and correctness. |
| Performance only | `--performance-only` | Compile and performance. |
| Torch compile reference | `--torch-compile` | Performance of `torch.compile(reference Model)`. |

Constraints:

- `--correctness-only` and `--performance-only` are mutually exclusive.
- `--torch-compile` cannot be combined with `--input`, `--correctness-only`, or `--performance-only`.
- `eval_mode=torch_compile_reference` always runs performance evaluation. If `validation_mode` is explicitly set in the config, it must be `performance_only`.

## 4. Recommended General Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `--atol` | float | `0.01` | Absolute tolerance, including near-zero rounding and cancellation residuals. |
| `--rtol` | float | `0.05` | Relative tolerance measured against the reference, not the candidate. |
| `--correctness-max-rel-l2` | float | Unset | Opt-in global relative L2 policy, replacing elementwise comparison. Mutually exclusive with `correctness_error_budgets`. |
| `--correctness-error-budgets` | JSON object | Unset | Per-output tolerances and optional RMS budgets; see below. |
| `--num-correctness-cases` | int | `1` | Original correctness draws per shape, or per explicit seed when `correctness_seeds` is set. Each enabled input profile adds a separate case per draw (Section 4.1). |
| `--warmup-iters` | int | `10` | Performance warmup budget. **In `eager` mode, the unit is milliseconds, not iterations** (the `warmup` argument to Triton's `do_bench`, documented as "Warmup time (in ms)"). In `cuda_graph_replay` mode, it is the number of replays. |
| `--bench-iters` | int | `100` | Performance benchmark budget. **In `eager` mode, the unit is milliseconds, not iterations** (the `rep` argument to `do_bench`, documented as "Repetition time (in ms)"). Thus, `--bench-iters 100` requests approximately 100 ms of measurement: a fast kernel may run thousands of times, while a slow kernel may run only once. The length of `samples` in `eval_result.json` gives the recorded sample count. In `cuda_graph_replay` mode, this option is the number of replays. The option name predates this distinction and is retained for compatibility. |
| `--candidate-timeout-s` | float | `60` | Timeout in seconds for candidate import, instantiation, and each correctness forward call; `<=0` disables it. |
| `--perf-timeout-s` | float | `600` | Timeout in seconds for the entire performance stage of each shape. In `torch_compile_reference` mode, it contributes to the wall-clock limit of each shape worker. `<=0` sets this budget to zero. |
| `--compile-timeout-s` | float | `300` | Independent wall-clock limit for the candidate compilation stage. If compilation does not finish within the budget, the entire **process group** receives SIGKILL so that compiler processes spawned by the worker cannot retain locks after being reparented. This budget also contributes to the overall worker limit; in `torch_compile_reference` mode, it contributes to each shape worker's wall-clock limit. |
| `--trust-mode` | enum | `trusted` | Allowed values: `trusted`, `untrusted`. |
| `--skip-kernel-attribution` | bool flag | `false` | Skips kernel attribution and `flydsl_compute_ratio`. |

On POSIX, each worker runs in its own process group. Timeout handling kills the
whole group, including compiler descendants; the deadline also covers inherited
stdout/stderr pipes after the worker exits. A supervising worker gets up to one
second to clean up its separately grouped shape worker before SIGKILL, followed
by bounded process/pipe cleanup. This is not a sandbox: descendants that explicitly
detach into another session are outside the original process group.

### Mixed Tolerance and Optional Output Error Budgets

The default floating-point policy is `mixed_with_optional_rms_v1`, recorded in
`runner_config.correctness_tolerance_policy`. Each finite reference element `r`
and candidate element `c` must satisfy:

```text
abs(c - r) <= atol + rtol * abs(r)
```

Absolute tolerance accepts near-zero rounding and cancellation residuals;
reference zeros do not require exact candidate zeros. Relative tolerance uses
the reference and accepts proportionate errors on large values. With defaults,
`0` versus `1e-8` passes in either direction, and `1e6` versus `1.001e6` also
passes. The computation uses scaling to avoid underflow/overflow. Maximum
absolute error is diagnostic, not a separate absolute-only rejection criterion.
Integer outputs still require exact equality.

Fixed absolute tolerance can also accept incorrect small outputs. Protect such
outputs explicitly with `correctness_error_budgets`; there is no universal noise
floor that distinguishes meaningful small signals from cancellation residuals.
CLI JSON, the SDK and `run_eval(...)` all accept this per-output mapping:

```json
{
  "correctness_error_budgets": {
    "values": {"rms_atol": 1e-7, "rms_rtol": 0.05, "dim": -1},
    "scales": {"atol": 0, "rtol": 0.05}
  }
}
```

Use the names in `correctness.shapes.<id>.cases[].outputs[].name`: `out` for a
single tensor, `scales` for a dictionary output, `output[0]` for a tuple output,
and dotted names for nested dictionaries. `"*"` supplies a default budget for
floating outputs; a specific output's object replaces the wildcard object in
full. Integer and `None` outputs ignore wildcard budgets; explicitly assigning
one to them is an error. Unknown output names fail visibly. Pass `{}` to disable
budgets from the CLI when overriding a config file.

| Budget field | Effect |
|---|---|
| `atol`, `rtol` | Override the corresponding global elementwise tolerance for this output. Omitted fields inherit global values. |
| `rms_atol`, `rms_rtol` | Enable an additional RMS guard. Both must be supplied to make the absolute noise allowance explicit. |
| `dim` | Optional integer axis for RMS reduction; every remaining group must pass. Omit or use `null` for whole-output RMS. For a 2-D tensor, `-1` checks each row independently. |

The RMS guard requires:

```text
RMS(candidate - reference) <= rms_atol + rms_rtol * RMS(reference)
```

Both elementwise comparison and an explicitly enabled RMS guard must pass.
RMS uses scaled float64 arithmetic. `max_rms_error_ratio` reports the maximum
ratio of measured RMS error to its allowed budget across groups; a finite value
above 1 fails. It is `null` when disabled or unrepresentable. A zero-budget
mismatch still fails; matching zero tensors have ratio 0.

For illustration, `rms_atol=1e-7, rms_rtol=0.05` accepts a `1e-8` cancellation
residual, but rejects an output filled with `1e-4` replaced by zero. These are
examples, not automatic defaults or universal recommendations. Signals below
the selected noise floor can be missed. Choose budgets from the operator's
contract, input scale, dtype and reference behavior. Use `dim` when a global
average could hide a bad row; no semantic blocks or reshapes are inferred.

`correctness_error_budgets` is mutually exclusive with
`correctness_max_rel_l2`, including its environment variable. The latter remains
an explicit alternative: `norm(c-r)/norm(r)` must meet its threshold. This strict
L2 mode has no absolute noise allowance or fixed denominator floor and can
reject near-zero cancellation differences. Two zero tensors have relative error
0; a zero reference versus a nonzero candidate fails strict L2. Choose output
RMS budgets when cancellation noise must be allowed.

Norms and cosine calculations use scaled float64 arithmetic.
`max_elementwise_rel_diff` uses actual reference magnitudes; undefined or
unrepresentable metrics are `null`, not JSON `Infinity`/`NaN`. Non-finite-output
rejection remains unchanged. CUDA Graph's default eager/replay check also uses
mixed tolerance; explicitly requested graph cosine/L2 policies retain their
separate semantics. Candidate output budgets apply only to the reference-versus-
candidate correctness stage, not graph replay checks.

### 4.1 Optional Seeds and Input Distributions

All options below are disabled by default. Omitting them preserves the existing
input factory, deterministic seeds, case artifacts, compilation,
and performance inputs. They apply only to candidate correctness evaluation;
setting them with `performance_only` or `torch_compile_reference` is an error.

Each distribution is an **independent top-level option** in the JSON config,
`atrex_bench.evaluate(config)` SDK, and `run_eval(...)` Python function. Its CLI
spelling replaces underscores with hyphens and takes a JSON value. For example:

```bash
python scripts/run_eval.py --config run_eval.json \
  --correctness-seeds '[0, 17, 42]' \
  --correctness-wide-uniform '{"inputs":["x"],"low":-16,"high":16}' \
  --correctness-zeros '{"inputs":["x"]}'
```

Use `false` or `null` in config to disable a distribution. To override an enabled
config distribution from the CLI, pass `false`, e.g. `--correctness-zeros false`.
CLI values replace the whole corresponding config object, not individual
parameters inside it.

| Config option | Parameters besides required `inputs` | Input coverage |
|---|---|---|
| `correctness_seeds` | Non-empty list of distinct integers in `[0, 2147483647]`; no `inputs` field | Explicit seeds for Python, NumPy and Torch RNGs. |
| `correctness_wide_uniform` | `low=-16`, `high=16` | Uniform floating-point values over a wide interval. |
| `correctness_sparse_outliers` | `density=0.01`, `magnitude=64` | Zero background with `ceil(numel * density)` randomly placed signed outliers (at least one). |
| `correctness_log_uniform` | `min_exponent=-4`, `max_exponent=4` | Random signs and magnitudes `10**U(min_exponent, max_exponent)`. |
| `correctness_cross_rank_cancellation` | `rank_dim=0`, `magnitude=256`, `residual=1` | Paired rank contributions with large opposite values and a small residual. |
| `correctness_zeros` | None | All-zero floating-point input. |
| `correctness_tiny_values` | `multiplier=1` | Alternating `±tiny * multiplier` and `±tiny * multiplier / 2`, where `tiny` is the target dtype's smallest positive normal value. |
| `correctness_nonlinear_saturation` | `magnitude=32` | Alternating positive/negative large values for nonlinear saturation. |
| `correctness_routing_all_ties` | `value=1` | Equal logits/scores along the last axis; at least two entries required. |
| `correctness_rounding_near_ties` | `rounding_dtype="bfloat16"` (or `"float16"`) | Values near 1 rounded into tied groups on adjacent representable levels; at least four entries on the last axis. |
| `correctness_signed_scale` | `magnitude=1` | Alternating negative/positive scales; a scalar tensor receives a negative scale. |
| `correctness_zero_scale` | None | All-zero scale tensors. |
| `correctness_fp4_extreme_codes` | None | Packed E2M1 `uint8` data: all 16 combinations of the `+0`, `-0`, `+6`, `-6` nibbles. |

`inputs` is a non-empty list of **forward argument paths**, e.g. `["x"]`,
`["a_scales", "b_scales"]`, or `["state.ranks.0"]`. Dot components select dictionary
keys or list/tuple indices. Top-level names come from the reference's `forward`
signature, including for positional legacy `get_inputs()` factories. Ordinary
profiles target floating-point tensors, including scalar tensors. No input names,
routing roles, integer index tensors or packed encodings are inferred. Select
only inputs whose operator contract permits that distribution. Initialization
arguments and Python scalar parameters are not modified.

Cross-rank cancellation accepts either a tensor whose `rank_dim` identifies the
rank contributions, or a list/tuple of tensors at the selected path. It requires
at least two contributions with matching shapes, dtypes and devices. Each pair
is generated as `a` and `-a + residual`, using the rounded `a`; an odd final rank
receives `residual`. This models the **input contributions** to a rank reduction;
it does not launch distributed processes. Configure a residual representable at
the selected magnitude and dtype.

Near-tie inputs use the repeating pre-rounding row pattern
`[1-eps/4, 1+eps/4, 1+3*eps/4, 1+5*eps/4]`, where `eps` belongs to
`rounding_dtype`. Round first, then store in the original input dtype. This makes
tied groups separated by one rounding ULP; targets coarser than `rounding_dtype`
are rejected. Select router scores/logits before top-k, not the resulting expert
IDs. Tie-breaking and output comparison still follow the existing reference.

FP4 profiles operate on two E2M1 nibbles per byte, not INT4 or unpacked floating
values. Nibbles `0x0`, `0x8`, `0x7`, `0xF` encode signed zeros and extrema. See
[NVIDIA's E2M1 value mapping](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py).
The pattern exercises both nibble positions and repeats every 16 bytes; shorter
tensors exercise only its prefix. Quantization scales are left as generated by
the original factory unless their separate scale profile is selected.

The runner regenerates fresh original inputs for every case, then applies exactly
one profile before cloning identical inputs for reference and candidate. It
preserves tensor shapes, dtypes, strides, devices and existing aliases. Non-target
inputs retain the factory's values (apart from aliases of an explicitly selected
target). Missing targets, unsupported dtypes, empty targets and overflowing
profile values produce a recorded correctness failure, never a silent pass.

For `N=num_correctness_cases`, `S=len(correctness_seeds)` (or `1` when unset),
and `P` enabled profiles, each shape has **`N * S * (1 + P)`** planned cases.
Every draw includes the original input case followed by the enabled profiles in
the table's order. Profiles are not composed with each other. For an explicit
seed, the first draw uses that seed verbatim; additional draws derive stable
seeds from it and their draw index. Without explicit seeds, the original
per-shape seed sequence is retained. Changing the set of enabled profiles does
not change baseline seeds. Factories must honor the seeded RNGs to reproduce
inputs. Worker timeouts account for the expanded case count.

Accuracy failures continue through the plan. Existing fatal-error/timeout abort
behavior remains; unexecuted cases are marked with an error and retain their
planned input identifiers. The comparison policy above applies to every case; input profiles do not
change its configured thresholds. Configure `atol`, `rtol`, or
`correctness_max_rel_l2` for the desired sensitivity. Performance timing always uses the original performance input path.

Example config for an operator with a floating input named `x`:

```json
{
  "schema_version": "v1",
  "input": "/abs/path/candidate.py",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "validation_mode": "correctness_only",
  "correctness_seeds": [0, 17, 42],
  "num_correctness_cases": 1,
  "correctness_wide_uniform": {"inputs": ["x"], "low": -16, "high": 16},
  "correctness_sparse_outliers": {"inputs": ["x"], "density": 0.01},
  "correctness_log_uniform": {"inputs": ["x"]},
  "correctness_zeros": {"inputs": ["x"]},
  "correctness_tiny_values": {"inputs": ["x"]}
}
```

This schedules 18 cases per shape. Scale, rank, routing and packed-data tests can
be enabled independently for operators exposing those input roles.

### 4.2 Replaying an Extended Correctness Case

`runner_config` records the enabled options with resolved profile defaults.
Baseline case artifacts remain `{"seed": ..., "format": "manual_seed"}`.
A transformed case uses this additive artifact representation:

```json
{
  "seed": 17,
  "format": "manual_seed_profile",
  "profile": "wide_uniform",
  "profile_version": 1,
  "parameters": {"inputs": ["x"], "low": -16, "high": 16}
}
```

To replay, use the archived reference/input bundle and the same runtime/device:

```python
import inspect
from atrex_bench.eval._runtime import load_shape_call_inputs, seed_all_input_rngs
from atrex_bench.eval.input_cases import InputCase, apply_input_case

# artifact is cases[i]["input_artifact"]; model, input_module, shape and device
# are loaded from the evaluation's archived reference bundle.
assert artifact["profile_version"] == 1
seed_all_input_rngs(artifact["seed"])
inputs = load_shape_call_inputs(input_module, shape, device)
case = InputCase(artifact["seed"], artifact["profile"], artifact["parameters"])
apply_input_case(inputs, inspect.signature(model.forward), case)
result = model(*inputs.args, **inputs.kwargs)
```

Replay must run the original input factory before applying the profile, since
both consume the seeded RNG sequence. No full tensor artifacts are stored.

## 5. Advanced Performance Options

| Option | Type | Default | Applies when |
|---|---:|---:|---|
| `--benchmark-mode` | enum | `eager` | Allowed values: `eager`, `cuda_graph_replay`. |
| `--cuda-graph-cache-flush-mb` | int | `1024` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-atol` | float | `0.01` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-rtol` | float | `0.05` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-min-cosine` | float | Unset | `benchmark-mode=cuda_graph_replay`. When set, replaces the default mixed-tolerance floating-point checks. |
| `--graph-max-rel-l2` | float | Unset | `benchmark-mode=cuda_graph_replay`. |

Unless `--skip-kernel-attribution` is set, CUDA Graph mode profiles a separate
loop of replays of the captured graph. Capture, eager warmup, and cache flushing
are excluded from this loop, and profiler overhead does not affect the recorded
end-to-end samples. Kernel times are summed across all launches of each symbol
per forward (or graph replay), not averaged per kernel launch.

## 6. GPU Clock-Locking Options

### 6.1 Clock-Lock Modes

| Option | Values | Default | Description |
|---|---|---|---|
| `--clock-lock-mode` | `off`, `external`, `manage` | `off` | GPU clock policy. |

The config field is `clock_lock_mode`. The following CLI options are retained only as compatibility aliases:

- `--lock-clocks`: equivalent to `--clock-lock-mode manage`.
- `--require-clock-locked`: equivalent to external-marker mode.
- `--clock-locked`: records only a caller assertion; it does not lock or verify hardware clocks.

### 6.2 Managed-Mode Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `--clock-lock-device` | string | Resolved automatically | Physical GPU index or GPU UUID. Explicitly passing a UUID is recommended on multi-GPU systems. |
| `--gpu-clock-mhz` | int | None | Required target graphics clock frequency. |
| `--memory-clock-mhz` | int | Unset | Set only when the device supports runtime memory-clock locking. |
| `--clock-lock-tolerance-mhz` | int | `50` | Initial verification tolerance after setting the clocks. Must be a positive integer. |
| `--clock-lock-settle-seconds` | float | `3.0` | Time to wait for clocks to settle after applying the settings. |
| `--clock-lock-command-timeout-s` | float | `10.0` | Timeout for each `nvidia-smi` command. |
| `--clock-lock-monitor` / `--no-clock-lock-monitor` | bool | Enabled | Whether to monitor the entire evaluation window. |
| `--clock-lock-sample-interval-ms` | int | `10` | Monitoring sample interval. Must be a positive integer. |
| `--clock-lock-runtime-tolerance-mhz` | int | `0` | Clock tolerance during the evaluation window. Must be a non-negative integer. |
| `--clock-lock-fail-on-deviation` / `--no-clock-lock-fail-on-deviation` | bool | `true` | Whether a clock deviation from the target causes the evaluation to fail. |
| `--allow-busy-gpu` | bool flag | `false` | Allows existing compute processes on the target GPU; busy GPUs are rejected by default. |

Managed-mode constraints:

- Only the top-level `run_eval` process may manage clock locking.
- `--gpu-clock-mhz` must be a positive integer.
- On multi-GPU systems, `CUDA_VISIBLE_DEVICES` and `--clock-lock-device` must identify the same physical GPU.
- Managed-mode options cannot be used in `off` or `external` mode.
- `manage` cannot be combined with `--clock-locked` or `--require-clock-locked`.

## 7. Configuration Rules

The Gateway exposes only `--config`. The JSON config supports all public capabilities in Sections 2-6. Use the following canonical fields for mode switches and compatibility aliases:

| CLI capability | Config field |
|---|---|
| `--torch-compile` | `"eval_mode": "torch_compile_reference"` |
| `--correctness-only` | `"validation_mode": "correctness_only"` |
| `--performance-only` | `"validation_mode": "performance_only"` |
| `--lock-clocks` | `"clock_lock_mode": "manage"` |
| `--allow-busy-gpu` | `"clock_lock_require_idle": false` |

`config_version` defaults to `v1` and is written to `runner_config.config_version` in `eval_result.json`. It has a different meaning from the top-level `schema_version`.

Configuration precedence:

```text
Explicit CLI options > JSON config > Built-in defaults
```

Example: full candidate evaluation:

```json
{
  "schema_version": "v1",
  "eval_mode": "candidate",
  "validation_mode": "full",
  "input": "/abs/path/candidate.py",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "warmup_iters": 10,
  "bench_iters": 100,
  "benchmark_mode": "eager",
  "trust_mode": "trusted"
}
```

Example: performance-only candidate evaluation with managed GPU clock locking:

```json
{
  "schema_version": "v1",
  "eval_mode": "candidate",
  "validation_mode": "performance_only",
  "input": "/abs/path/candidate.py",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "clock_lock_mode": "manage",
  "clock_lock_device": "GPU-01234567-89ab-cdef-0123-456789abcdef",
  "gpu_clock_mhz": 2000,
  "clock_lock_monitor": true,
  "clock_lock_sample_interval_ms": 10,
  "clock_lock_runtime_tolerance_mhz": 5,
  "clock_lock_fail_on_deviation": false,
  "clock_lock_require_idle": true
}
```

## 8. Private Options: Do Not Expose

The following options are reserved for communication between `run_eval` parent and child processes:

- `--worker`
- `--torch-compile-worker`
- `--single-shape-worker`
- `--torch-compile-shape-worker`
- `--artifact-dir`
- `--checkpoint-root`
- `--shape-id`
- `--shape-result-output`
- `--sdk-result-path-output`

`--sdk-result-path-output` is used only to pass the absolute path of the final
`eval_result.json` back to the SDK parent process. It is not a public option for
the Gateway or regular users.

## 9. Standard Launch Commands

### 9.1 Standard Gateway Entry Point

```bash
python scripts/run_eval.py --config /abs/path/run_eval.json
```

Correctness-only, performance-only, full, Torch compile, and clock-locking modes are all controlled through JSON fields. No additional Gateway options are needed.

### 9.2 Torch Compile Configuration

```json
{
  "schema_version": "v1",
  "eval_mode": "torch_compile_reference",
  "validation_mode": "performance_only",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "warmup_iters": 10,
  "bench_iters": 100,
  "compile_timeout_s": 300,
  "perf_timeout_s": 600
}
```

### 9.3 Existing CLI Compatibility

```bash
python scripts/run_eval.py \
  --input /abs/path/candidate.py \
  --reference-dir /abs/path/operator \
  --output /abs/path/results \
  --performance-only
```

## 10. Exit Codes and Artifacts

| Exit code | Meaning |
|---:|---|
| `0` | All stages required by the selected mode passed. |
| `1` | Evaluation failure, runtime error, or configuration validation failure. |
| `2` | An argparse error involving argument syntax, required options, enum values, or mutually exclusive options. |

Standard output:

```text
[OUTPUT] <output>/<timestamp>/<operator>/eval_result.json
```

Main artifacts:

- `<output>/<timestamp>/<operator>/eval_result.json`
- `<output>/<timestamp>/<operator>/staging_manifest.json`
- `clock_lock.json` and `clock_lock_trace.csv` in managed clock-locking mode

Progress logs are written to stderr. Integrations must use both the process exit code and `eval_result.json` to determine the outcome.
