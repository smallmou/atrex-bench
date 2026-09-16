"""Stage 1: Correctness verification against the eager reference baseline."""

from __future__ import annotations

import inspect
import json
import math
import os
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from atrex_bench.eval._runtime import (
    ShapeSpec,
    clone_model_inputs,
    flatten_outputs,
    get_device,
    instantiate_model_pair,
    load_reference_inputs,
    load_shape_call_inputs,
    load_shape_spec,
    seed_all_input_rngs,
    sync_device,
    write_input_artifact,  # noqa: F401 - backward-compatible module export
)
from atrex_bench.eval._timeout import CandidateTimeoutError, candidate_timeout
from atrex_bench.eval.input_cases import CorrectnessInputConfig, apply_input_case
from atrex_bench.eval.reward_hack import (
    RewardHackDetected,
    check_plain_tensor_outputs,
)

_DEFAULT_CANDIDATE_TIMEOUT_S = 60
CORRECTNESS_MAX_REL_L2_ENV = "ATREX_CORRECTNESS_MAX_REL_L2"
CORRECTNESS_TOLERANCE_POLICY = "mixed_with_optional_rms_v1"


@dataclass(frozen=True)
class OutputDiff:
    """Per-output comparison result for one correctness case.

    Fields match the data schema spec, Section 7 outputs entry exactly:
    ``name`` / ``passed`` / ``max_elementwise_abs_diff`` /
    ``max_elementwise_rel_diff`` / ``error``. dtype / shape are intentionally
    not recorded — they are derivable from metadata.json.output_dtypes and
    do not have a real consumer.
    """

    name: str
    passed: bool
    max_elementwise_abs_diff: float | None = None
    max_elementwise_rel_diff: float | None = None
    relative_l2: float | None = None
    max_rms_error_ratio: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CorrectnessCase:
    """One correctness case: a single random input draw.

    ``input_artifact`` records the seed and, for opt-in distributions, the
    versioned profile and parameters. Replay uses shapes.json + input.py;
    full tensor payloads are not persisted.
    """

    input_artifact: dict[str, object] | None
    outputs: list[OutputDiff] = field(default_factory=list)
    mutated_inputs: list[OutputDiff] = field(default_factory=list)
    unexpected_mutations: list[OutputDiff] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class CorrectnessShapeResult:
    """Per-shape correctness result returned by ``check_correctness``.

    ``status`` ∈ ``{"passed", "failed", "skipped"}`` is what bubbles up to
    ``eval_result.json.passed.correctness.<shape_id>``; ``cases`` populate
    ``eval_result.json.correctness.shapes.<shape_id>.cases``.
    ``check_correctness`` itself only returns ``passed`` or ``failed``;
    ``skipped`` is set by run_eval when the stage was not run because of an
    earlier-stage failure.
    """

    status: str
    reason: str | None = None
    cases: list[CorrectnessCase] = field(default_factory=list)


def _is_leaf_output(value: object) -> bool:
    """Return whether a value is a leaf in the output tree (tensor or numeric scalar)."""
    return value is None or isinstance(value, (torch.Tensor, bool, int, float))


def _describe_output_structure(value: object) -> str:
    """Return a short structural label for one output node, used in error messages."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())!r})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def _validate_output_structures_match(
    reference: object,
    candidate: object,
    *,
    path: str = "output",
    strict_types: bool = False,
) -> None:
    """Raise ValueError if reference / candidate output structures don't match."""
    if (strict_types and type(reference) is not type(candidate)) or (
        (reference is None) != (candidate is None)
    ):
        raise ValueError(
            f"Output structure mismatch at {path}: reference={type(reference).__name__}, "
            f"candidate={type(candidate).__name__}"
        )
    ref_is_dict = isinstance(reference, dict)
    cand_is_dict = isinstance(candidate, dict)
    if ref_is_dict != cand_is_dict:
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}. "
            "When the reference Model.forward returns dict[str, Tensor], the "
            "candidate must also return a dict with matching keys."
        )
    if ref_is_dict:
        ref_keys = set(reference.keys())
        cand_keys = set(candidate.keys())
        if ref_keys != cand_keys:
            missing_in_candidate = sorted(ref_keys - cand_keys)
            extra_in_candidate = sorted(cand_keys - ref_keys)
            raise ValueError(
                f"Output structure mismatch at {path}: dict keys differ. "
                f"reference keys={sorted(ref_keys)}, "
                f"candidate keys={sorted(cand_keys)}, "
                f"missing in candidate={missing_in_candidate}, "
                f"extra in candidate={extra_in_candidate}."
            )
        for key in sorted(ref_keys):
            _validate_output_structures_match(
                reference[key], candidate[key], path=f"{path}.{key}", strict_types=strict_types
            )
        return

    ref_is_seq = isinstance(reference, (tuple, list))
    cand_is_seq = isinstance(candidate, (tuple, list))
    if ref_is_seq != cand_is_seq:
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}."
        )
    if ref_is_seq:
        if len(reference) != len(candidate):
            raise ValueError(
                f"Output structure mismatch at {path}: sequence length differs. "
                f"reference={len(reference)}, candidate={len(candidate)}."
            )
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _validate_output_structures_match(
                ref_item, cand_item, path=f"{path}[{index}]", strict_types=strict_types
            )
        return

    if not (_is_leaf_output(reference) and _is_leaf_output(candidate)):
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}."
        )


def _flatten_output_name(prefix_path: str) -> str:
    """Convert flatten_outputs() prefix paths to the schema's tensor name.

    flatten_outputs emits "output" for single tensors, "output.<key>" for dict
    branches, "output[i]" for tuple branches. The schema records just the
    tensor name from metadata.json.output_dtypes:
      "output"        -> "out"           (single-tensor convention)
      "output.<key>"  -> "<key>"         (dict tensor name)
      "output[i]"     -> kept as-is      (legacy tuple path; not reachable
                                          under the current schema, but
                                          tolerated for backward compat)
    """
    if prefix_path == "output":
        return "out"
    if prefix_path.startswith("output."):
        return prefix_path[len("output.") :]
    return prefix_path


def configured_max_rel_l2(explicit: float | None = None) -> float | None:
    if explicit is not None:
        value = float(explicit)
    else:
        raw = os.environ.get(CORRECTNESS_MAX_REL_L2_ENV)
        if raw is None or not raw.strip():
            return None
        value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("correctness max_rel_l2 must be finite and non-negative")
    return value


def _finite_metric(value: float) -> float | None:
    """Keep undefined/unrepresentable error metrics JSON-compatible."""
    return value if math.isfinite(value) else None


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    """Relative L2 for finite float64 tensors without a magnitude-dependent floor.

    Normalize before taking norms to avoid squaring tiny/large values. A zero
    reference with a nonzero candidate has undefined relative error (None),
    which fails any relative-L2 threshold.
    """
    if reference.numel() == 0:
        return 0.0
    scale = torch.maximum(reference.abs().max(), candidate.abs().max())
    if scale.item() == 0:
        return 0.0
    reference_scaled = reference / scale
    candidate_scaled = candidate / scale
    reference_norm = torch.linalg.vector_norm(reference_scaled)
    if reference_norm.item() == 0:
        return None
    difference_norm = torch.linalg.vector_norm(candidate_scaled - reference_scaled)
    return _finite_metric(float((difference_norm / reference_norm).item()))


def validate_error_budgets(value: dict | None) -> dict | None:
    """Validate per-output tolerances; an RMS guard requires an explicit noise budget."""
    if value is None or value == {}:
        return None
    if not isinstance(value, dict):
        raise ValueError("correctness_error_budgets must be an object keyed by output name")
    normalized = {}
    for name, budget in value.items():
        if not isinstance(name, str) or not name or not isinstance(budget, dict) or not budget:
            raise ValueError("Each correctness error budget needs an output name and parameters")
        unknown = budget.keys() - {"atol", "rtol", "rms_atol", "rms_rtol", "dim"}
        if unknown:
            raise ValueError(f"{name}: unknown error budget parameters: {sorted(unknown)}")
        has_rms = "rms_atol" in budget or "rms_rtol" in budget
        if has_rms and not {"rms_atol", "rms_rtol"} <= budget.keys():
            raise ValueError(f"{name}: specify both rms_atol and rms_rtol")
        if "dim" in budget and not has_rms:
            raise ValueError(f"{name}: dim requires an RMS budget")
        for key, number in budget.items():
            if key == "dim":
                if number is not None and type(number) is not int:
                    raise ValueError(f"{name}.dim must be an integer or null")
            elif type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError(f"{name}.{key} must be finite and non-negative")
        normalized[name] = dict(budget)
    return normalized


def _rms_budget_check(
    reference: torch.Tensor, candidate: torch.Tensor, budget: dict
) -> tuple[bool, float | None]:
    """Check RMS(error) <= rms_atol + rms_rtol * RMS(reference), per group.

    With dim=None the group is the whole output; otherwise reduce along that
    axis and require every remaining group to pass. Scale before squaring so
    the absolute noise budget works for tiny and large outputs alike.
    """
    dim = budget.get("dim")
    if dim is not None and not -reference.ndim <= dim < reference.ndim:
        raise ValueError("RMS budget dim is outside output dimensions")
    if reference.numel() == 0:
        return True, 0.0
    scale = torch.maximum(
        reference.abs().amax(dim=dim, keepdim=True),
        candidate.abs().amax(dim=dim, keepdim=True),
    )
    scale = torch.where(scale == 0, 1.0, scale)
    reference_scaled = reference / scale
    candidate_scaled = candidate / scale
    error_rms = (candidate_scaled - reference_scaled).square().mean(dim=dim, keepdim=True).sqrt()
    reference_rms = reference_scaled.square().mean(dim=dim, keepdim=True).sqrt()
    limit = budget["rms_atol"] / scale + budget["rms_rtol"] * reference_rms
    ratio = torch.where(error_rms == 0, 0.0, error_rms / limit)
    return bool((error_rms <= limit).all().item()), _finite_metric(float(ratio.max().item()))


def _compare_output_tensors(
    reference_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
    max_rel_l2: float | None = None,
    strict_dtype: bool = False,
    error_budget: dict | None = None,
) -> OutputDiff:
    """Compare a pair of output tensors and return the per-output diff record."""
    if strict_dtype and reference_tensor.dtype != candidate_tensor.dtype:
        return OutputDiff(
            name=name,
            passed=False,
            error=f"Output dtype mismatch: {reference_tensor.dtype} != {candidate_tensor.dtype}",
        )
    if reference_tensor.shape != candidate_tensor.shape:
        return OutputDiff(
            name=name,
            passed=False,
            error=(
                "Output shape mismatch: "
                f"reference={list(reference_tensor.shape)}, "
                f"candidate={list(candidate_tensor.shape)}"
            ),
        )

    error: str | None = None
    max_rms_error_ratio = None
    if error_budget is not None:
        if max_rel_l2 is not None:
            raise ValueError("correctness_error_budgets cannot be combined with max_rel_l2")
        if not torch.is_floating_point(reference_tensor):
            raise ValueError(f"Error budget for {name} requires a floating-point reference output")
        atol = error_budget.get("atol", atol)
        rtol = error_budget.get("rtol", rtol)
    if torch.is_floating_point(reference_tensor) or torch.is_floating_point(candidate_tensor):
        reference_float = reference_tensor.detach().to(torch.float64)
        candidate_float = candidate_tensor.detach().to(torch.float64)
        ref_is_finite = bool(torch.isfinite(reference_float).all().item())
        cand_is_finite = bool(torch.isfinite(candidate_float).all().item())
        if not ref_is_finite or not cand_is_finite:
            return OutputDiff(
                name=name,
                passed=False,
                max_elementwise_abs_diff=0.0,
                max_elementwise_rel_diff=0.0,
                relative_l2=0.0,
                error=(
                    "Non-finite output detected: "
                    f"reference_finite={ref_is_finite}, candidate_finite={cand_is_finite}"
                ),
            )

        reference_abs = reference_float.abs()
        reference_nonzero = reference_abs != 0
        candidate_is_zero = bool(
            reference_nonzero.any().item() and not candidate_float.ne(0).any().item()
        )
        abs_diff = (candidate_float - reference_float).abs()
        max_elementwise_abs_diff = (
            _finite_metric(float(abs_diff.max().item())) if abs_diff.numel() else 0.0
        )
        relative_errors = abs_diff / reference_abs
        # Opposite-sign finite values can overflow subtraction even though
        # their relative error (e.g. 2) is representable.
        relative_errors = torch.where(
            torch.isinf(abs_diff) & reference_nonzero,
            (candidate_float / reference_float - 1).abs(),
            relative_errors,
        )
        relative_errors = torch.where(
            reference_nonzero,
            relative_errors,
            torch.where(candidate_float == 0, 0.0, float("inf")),
        )
        max_elementwise_rel_diff = (
            _finite_metric(float(relative_errors.max().item())) if abs_diff.numel() else 0.0
        )
        del relative_errors, reference_nonzero
        relative_l2 = _relative_l2(reference_float, candidate_float)
        if max_rel_l2 is not None:
            passed = relative_l2 is not None and relative_l2 <= max_rel_l2
        else:
            # Normalize each element's mixed tolerance to avoid overflow. A
            # tiny reference versus a huge candidate must not become inf <= inf.
            scale = torch.maximum(reference_abs, candidate_float.abs())
            scale = torch.where(scale == 0, 1.0, scale)
            normalized_error = torch.where(
                torch.isinf(abs_diff),
                (candidate_float / scale - reference_float / scale).abs(),
                abs_diff / scale,
            )
            limit = atol / scale + rtol * (reference_abs / scale)
            passed = bool((normalized_error <= limit).all().item())
            # Large carried states can occupy gigabytes. These elementwise
            # intermediates are no longer needed when the RMS reduction starts.
            del reference_abs, abs_diff, scale, normalized_error, limit
            if error_budget is not None and "rms_atol" in error_budget:
                rms_passed, max_rms_error_ratio = _rms_budget_check(
                    reference_float, candidate_float, error_budget
                )
                passed = passed and rms_passed
        if candidate_is_zero and not passed:
            error = "Candidate output is all zero while reference output is non-zero"
    else:
        passed = bool(torch.equal(reference_tensor, candidate_tensor))
        max_elementwise_abs_diff = 0.0 if passed else 1.0
        max_elementwise_rel_diff = 0.0 if passed else 1.0
        relative_l2 = None

    return OutputDiff(
        name=name,
        passed=passed,
        max_elementwise_abs_diff=max_elementwise_abs_diff,
        max_elementwise_rel_diff=max_elementwise_rel_diff,
        relative_l2=relative_l2,
        max_rms_error_ratio=max_rms_error_ratio,
        error=error,
    )


def _compare_value_trees(
    reference, candidate, *, name, atol, rtol, max_rel_l2=None, strict_types=False,
    error_budgets=None,
):
    """Compare return values, with exact types for explicit mutation contracts."""
    try:
        _validate_output_structures_match(
            reference, candidate, path=name, strict_types=strict_types
        )
    except ValueError as error:
        return [OutputDiff(name=name, passed=False, error=str(error))]
    if isinstance(reference, dict):
        return [
            diff
            for key in reference
            for diff in _compare_value_trees(
                reference[key],
                candidate[key],
                name=f"{name}.{key}",
                atol=atol,
                rtol=rtol,
                max_rel_l2=max_rel_l2,
                strict_types=strict_types,
                error_budgets=error_budgets,
            )
        ]
    if isinstance(reference, (list, tuple)):
        return [
            diff
            for index, (left, right) in enumerate(zip(reference, candidate))
            for diff in _compare_value_trees(
                left,
                right,
                name=f"{name}[{index}]",
                atol=atol,
                rtol=rtol,
                max_rel_l2=max_rel_l2,
                strict_types=strict_types,
                error_budgets=error_budgets,
            )
        ]
    budgets = error_budgets or {}
    output_name = _flatten_output_name(name)
    budget = budgets.get(output_name, budgets.get("*"))
    if reference is None and output_name in budgets:
        raise ValueError(f"Error budget for {name} requires a floating-point reference output")
    if reference is None:
        return [OutputDiff(name=name, passed=True)]
    # Retain legacy scalar conversion and tolerance semantics as well as
    # list/tuple interoperability for benchmarks without an explicit contract.
    reference_tensor = flatten_outputs(reference)[0][1]
    candidate_tensor = flatten_outputs(candidate)[0][1]
    if not torch.is_floating_point(reference_tensor) and output_name not in budgets:
        budget = None  # Wildcard budgets cover floating outputs; integers stay exact.
    return [
        _compare_output_tensors(
            reference_tensor,
            candidate_tensor,
            name=name,
            atol=atol,
            rtol=rtol,
            max_rel_l2=max_rel_l2,
            strict_dtype=strict_types,
            error_budget=budget,
        )
    ]


def _check_unchanged(before, after, *, name):
    """Check full backing bytes (including pool tails) without float tolerances."""
    if type(before) is not type(after):
        return [OutputDiff(name=name, passed=False, error="Input type changed")]
    if isinstance(before, torch.Tensor):
        same = (
            before.dtype == after.dtype
            and before.shape == after.shape
            and before.stride() == after.stride()
            and before.storage_offset() == after.storage_offset()
        )

        def raw(tensor):
            storage = tensor.untyped_storage()
            return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )

        same = same and torch.equal(raw(before), raw(after))
        return [
            OutputDiff(
                name=name, passed=bool(same), error=None if same else "Undeclared input mutation"
            )
        ]
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return [OutputDiff(name=name, passed=False, error="Input keys changed")]
        return [
            diff
            for key in before
            for diff in _check_unchanged(before[key], after[key], name=f"{name}.{key}")
        ]
    if isinstance(before, (tuple, list)):
        if len(before) != len(after):
            return [OutputDiff(name=name, passed=False, error="Input length changed")]
        return [
            diff
            for index, (left, right) in enumerate(zip(before, after))
            for diff in _check_unchanged(left, right, name=f"{name}[{index}]")
        ]
    same = before == after
    return [OutputDiff(name=name, passed=bool(same), error=None if same else "Input value changed")]


def check_correctness(
    reference_path: Path,
    candidate_path: Path,
    *,
    shape_id: str = "0",
    atol: float = 1e-2,
    rtol: float = 0.05,
    num_correctness_cases: int = 1,
    device: str = "auto",
    artifact_dir: Path | None = None,
    artifact_root: Path | None = None,
    candidate_timeout_s: int | float | None = _DEFAULT_CANDIDATE_TIMEOUT_S,
    max_rel_l2: float | None = None,
    untrusted_mode: bool = False,
    correctness_inputs: CorrectnessInputConfig | None = None,
    correctness_error_budgets: dict | None = None,
) -> CorrectnessShapeResult:
    """Compare candidate outputs against the eager reference baseline for one shape.

    ``shape_id`` selects the entry from ``shapes.json`` next to ``reference_path``.
    Synthetic inline references without a sibling shapes.json fall back to the
    legacy ``get_inputs()`` / ``get_init_inputs()`` path; ``shape_id`` is then
    informational only.
    """
    if num_correctness_cases < 1:
        return CorrectnessShapeResult(
            status="failed",
            reason="num_correctness_cases must be at least 1",
        )
    try:
        effective_max_rel_l2 = configured_max_rel_l2(max_rel_l2)
        correctness_error_budgets = validate_error_budgets(correctness_error_budgets)
        if correctness_error_budgets and effective_max_rel_l2 is not None:
            raise ValueError("correctness_error_budgets cannot be combined with max_rel_l2")
    except ValueError as error:
        return CorrectnessShapeResult(status="failed", reason=str(error))

    try:
        resolved_device = get_device(device)
        loaded_models = instantiate_model_pair(
            reference_path,
            candidate_path,
            resolved_device,
            module_prefix="atrex_correctness",
            shape_id=shape_id,
            candidate_timeout_s=candidate_timeout_s,
        )
        shape: ShapeSpec | None
        if (reference_path.parent / "shapes.json").is_file():
            shape = load_shape_spec(reference_path, shape_id)
        else:
            shape = None
        metadata_path = reference_path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        contract = metadata.get("benchmark_contract", {})
        mutations = contract.get("mutates_inputs", [])
        strict_types = "mutates_inputs" in contract
        if not isinstance(mutations, list) or not all(isinstance(x, str) for x in mutations):
            raise ValueError("benchmark_contract.mutates_inputs must be a list of input names")
        signature = inspect.signature(loaded_models.reference_model.forward)
    except Exception:
        return CorrectnessShapeResult(
            status="failed",
            reason=traceback.format_exc(),
        )

    case_plan = (correctness_inputs or CorrectnessInputConfig()).cases(
        shape_id, num_correctness_cases
    )
    total_cases = len(case_plan)
    case_records: list[CorrectnessCase] = []
    failed_cases = 0
    # Surfaced into CorrectnessShapeResult.reason for the deterministic-failure
    # paths, so the shape-level summary (which is what bubbles up to the
    # eval_result.json passed.correctness.<id>.reason field) says *why* the
    # shape failed, not just "X/N cases failed".
    early_abort_reason: str | None = None

    def _abort_remaining_cases(after_case_index: int, short_reason: str) -> None:
        """Append 'skipped' CorrectnessCase entries for every case after this one.

        Used for *non-accuracy* failures only — timeouts, exceptions during the
        candidate call, structural / shape / count mismatches. All of these are
        deterministic w.r.t. the candidate (same code + same inputs class -> same
        failure), so re-running with a fresh random draw of inputs is pure
        wasted wall-clock. Accuracy diffs (atol/rtol failures) deliberately do
        NOT trigger early-abort: those CAN be input-dependent and we want full
        per-case coverage to surface the diff distribution.

        ``short_reason`` is ALSO bubbled up into the shape-level
        ``CorrectnessShapeResult.reason`` so the eval_result.json
        ``passed.correctness.<id>.reason`` field tells you *why* the shape
        failed instead of just "X/N cases failed".
        """
        nonlocal failed_cases, early_abort_reason
        early_abort_reason = short_reason
        skipped = total_cases - (after_case_index + 1)
        if skipped <= 0:
            return
        failed_cases += skipped
        reason = f"skipped after case {after_case_index} failed deterministically: {short_reason}"
        for planned_case in case_plan[after_case_index + 1:]:
            artifact = planned_case.artifact() if correctness_inputs is not None else None
            case_records.append(CorrectnessCase(input_artifact=artifact, error=reason))

    for case_index, input_case in enumerate(case_plan):
        artifact = input_case.artifact()
        failure_stage = "input generation/profile"
        try:
            seed_all_input_rngs(input_case.seed)
            if shape is not None:
                inputs = load_shape_call_inputs(loaded_models.input_module, shape, resolved_device)
            else:
                inputs = load_reference_inputs(loaded_models.input_module, resolved_device)
            if input_case.profile is not None:
                apply_input_case(inputs, signature, input_case)
            failure_stage = "candidate"
            reference_call_inputs = clone_model_inputs(inputs)
            candidate_call_inputs = clone_model_inputs(inputs)

            def named(call):
                return signature.bind(*call.args, **call.kwargs).arguments

            original_named = named(inputs)
            if set(mutations) - original_named.keys():
                raise ValueError(
                    f"Unknown mutated input names: {set(mutations) - original_named.keys()}"
                )
            with torch.inference_mode():
                # Reference is the golden implementation; we trust it and
                # never time it out. The candidate is the AI-generated code
                # and may hang in JIT compile or run a pathological kernel,
                # so we wrap its call (plus the trailing GPU sync so the
                # alarm fires while we are still inside the scope).
                reference_output = loaded_models.reference_model(
                    *reference_call_inputs.args,
                    **reference_call_inputs.kwargs,
                )
                try:
                    with candidate_timeout(candidate_timeout_s):
                        candidate_output = loaded_models.candidate_model(
                            *candidate_call_inputs.args,
                            **candidate_call_inputs.kwargs,
                        )
                        sync_device(resolved_device)
                except CandidateTimeoutError as timeout_error:
                    failed_cases += 1
                    case_records.append(
                        CorrectnessCase(
                            input_artifact=artifact,
                            error=str(timeout_error),
                        )
                    )
                    _abort_remaining_cases(case_index, f"{candidate_timeout_s}s timeout")
                    break
                if untrusted_mode:
                    try:
                        check_plain_tensor_outputs(candidate_output)
                    except RewardHackDetected as reward_hack:
                        failed_cases += 1
                        case_records.append(
                            CorrectnessCase(
                                input_artifact=artifact,
                                error=str(reward_hack),
                            )
                        )
                        _abort_remaining_cases(case_index, "untrusted output guard failed")
                        break

            try:
                _validate_output_structures_match(
                    reference_output, candidate_output, strict_types=strict_types
                )
            except ValueError as structure_error:
                failed_cases += 1
                case_records.append(
                    CorrectnessCase(
                        input_artifact=artifact,
                        error=str(structure_error),
                    )
                )
                _abort_remaining_cases(case_index, "output structure mismatch")
                break

            failure_stage = "output comparison"
            output_diffs = _compare_value_trees(
                reference_output,
                candidate_output,
                name="output",
                atol=atol,
                rtol=rtol,
                max_rel_l2=effective_max_rel_l2,
                strict_types=strict_types,
                error_budgets=correctness_error_budgets,
            )
            output_diffs = [
                replace(diff, name=_flatten_output_name(diff.name)) for diff in output_diffs
            ]
            if correctness_error_budgets:
                unknown = correctness_error_budgets.keys() - {"*"} - {
                    diff.name for diff in output_diffs
                }
                if unknown:
                    raise ValueError(
                        f"Unknown correctness output budget target(s): {sorted(unknown)}"
                    )
            reference_named = named(reference_call_inputs)
            candidate_named = named(candidate_call_inputs)
            mutation_diffs = []
            unexpected_diffs = []
            for key, before in original_named.items():
                if key in mutations:
                    mutation_diffs.extend(
                        _compare_value_trees(
                            reference_named[key],
                            candidate_named[key],
                            name=f"input.{key}",
                            atol=atol,
                            rtol=rtol,
                            max_rel_l2=effective_max_rel_l2,
                            strict_types=True,
                        )
                    )
                else:
                    for role, state in (
                        ("reference", reference_named),
                        ("candidate", candidate_named),
                    ):
                        unexpected_diffs.extend(
                            _check_unchanged(before, state[key], name=f"{role}.input.{key}")
                        )
            all_diffs = output_diffs + mutation_diffs + unexpected_diffs
            case_passed = all(diff.passed for diff in all_diffs)
            has_structural_failure = any(diff.error is not None for diff in all_diffs)

            if not case_passed:
                failed_cases += 1

            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    outputs=output_diffs,
                    mutated_inputs=mutation_diffs,
                    unexpected_mutations=unexpected_diffs,
                )
            )

            if has_structural_failure:
                _abort_remaining_cases(case_index, "per-tensor shape/structural mismatch")
                break
            # Do not keep the previous case's input/state copies alive while
            # allocating the next seed/profile. Diff records contain only scalars.
            del inputs, reference_call_inputs, candidate_call_inputs
            del reference_output, candidate_output
            del original_named, reference_named, candidate_named
            before = state = None
        except Exception:
            # Any other exception in the case body (e.g. candidate raised an
            # OOM, kernel launch error, AttributeError on .Model, etc.) is
            # also deterministic w.r.t. the candidate. Record this case as
            # failed AND abort the remaining cases.
            failed_cases += 1
            tb = traceback.format_exc()
            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    error=tb,
                )
            )
            _abort_remaining_cases(case_index, f"{failure_stage} raised exception")
            break

    if failed_cases == 0:
        status = "passed"
        reason: str | None = None
    else:
        status = "failed"
        base = f"{failed_cases}/{total_cases} correctness cases failed"
        if early_abort_reason is not None:
            reason = f"{base}: {early_abort_reason}"
        else:
            reason = base

    return CorrectnessShapeResult(
        status=status,
        reason=reason,
        cases=case_records,
    )
