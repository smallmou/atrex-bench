"""Tests for Stage 1: correctness verification."""

from pathlib import Path

from atrex_bench.eval.correctness import check_correctness

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "references" / "atrex_001" / "reference.py"
CANDIDATE_PATH = Path(__file__).parent / "fixtures" / "generations" / "atrex_001.py"


def _write_python_file(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _all_outputs_passed(case) -> bool:
    return all(diff.passed for diff in case.outputs)


def _passed_case_count(result) -> int:
    return sum(
        1
        for case in result.cases
        if case.error is None and case.outputs and _all_outputs_passed(case)
    )


def test_correct_output_passes_with_multiple_cases() -> None:
    result = check_correctness(
        REFERENCE_PATH,
        CANDIDATE_PATH,
        num_correctness_cases=2,
        rtol=0.05,
        device="cpu",
    )
    assert result.status == "passed"
    assert result.reason is None
    assert len(result.cases) == 2
    assert _passed_case_count(result) == 2
    for case in result.cases:
        assert case.error is None
        for diff in case.outputs:
            assert diff.passed is True
            assert diff.max_elementwise_abs_diff == 0.0


def test_relative_tolerance_is_configurable(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return x + 1.0",
                "",
                "def get_inputs():",
                "    return [torch.zeros(4, 4)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return (x + 1.0) * 1.03",
            ]
        ),
    )

    loose_result = check_correctness(
        reference_path,
        candidate_path,
        rtol=0.05,
        device="cpu",
    )
    strict_result = check_correctness(
        reference_path,
        candidate_path,
        rtol=0.01,
        device="cpu",
    )

    assert loose_result.status == "passed"
    assert strict_result.status == "failed"
    strict_max_rel = strict_result.cases[0].outputs[0].max_elementwise_rel_diff
    assert strict_max_rel is not None
    assert strict_max_rel > 0.01


def test_relative_l2_policy_is_configurable(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return torch.ones_like(x)",
                "",
                "def get_inputs():",
                "    return [torch.zeros(4, 4)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return torch.full_like(x, 1.1)",
            ]
        ),
    )

    accepted = check_correctness(
        reference_path,
        candidate_path,
        max_rel_l2=0.2,
        device="cpu",
    )
    rejected = check_correctness(
        reference_path,
        candidate_path,
        max_rel_l2=0.05,
        device="cpu",
    )

    assert accepted.status == "passed"
    assert rejected.status == "failed"
    relative_l2 = accepted.cases[0].outputs[0].relative_l2
    assert relative_l2 is not None
    assert abs(relative_l2 - 0.1) < 1e-6


def test_nonfinite_candidate_output_fails_with_finite_metrics(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.ones_like(x)",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.full_like(x, float('nan'))",
            ]
        ),
    )

    result = check_correctness(reference_path, candidate_path, device="cpu")

    assert result.status == "failed"
    diff = result.cases[0].outputs[0]
    assert diff.passed is False
    assert diff.error is not None
    assert "Non-finite output" in diff.error
    assert diff.max_elementwise_abs_diff == 0.0
    assert diff.max_elementwise_rel_diff == 0.0
    assert diff.relative_l2 == 0.0


def test_all_zero_candidate_output_fails(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.ones_like(x)",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.zeros_like(x)",
            ]
        ),
    )

    result = check_correctness(reference_path, candidate_path, device="cpu")

    assert result.status == "failed"
    diff = result.cases[0].outputs[0]
    assert diff.passed is False
    assert diff.error == "Candidate output is all zero while reference output is non-zero"
    assert diff.max_elementwise_rel_diff == 1.0
    assert diff.relative_l2 == 1.0


def test_all_zero_candidate_respects_configured_tolerance(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.full_like(x, 1e-6)",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.zeros_like(x)",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        atol=1e-2,
        rtol=0.05,
        device="cpu",
    )

    assert result.status == "passed"
    diff = result.cases[0].outputs[0]
    assert diff.passed is True
    assert diff.error is None


def test_integer_mismatch_records_finite_rel_diff(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.ones(2, 2, dtype=torch.int64)",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.zeros(2, 2, dtype=torch.int64)",
            ]
        ),
    )

    result = check_correctness(reference_path, candidate_path, device="cpu")

    assert result.status == "failed"
    diff = result.cases[0].outputs[0]
    assert diff.max_elementwise_abs_diff == 1.0
    assert diff.max_elementwise_rel_diff == 1.0
    assert diff.relative_l2 is None
    assert diff.error is None


def test_runtime_error_fails(tmp_path: Path) -> None:
    candidate_path = _write_python_file(
        tmp_path,
        "runtime_error.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        raise RuntimeError('intentional failure')",
            ]
        ),
    )

    result = check_correctness(
        REFERENCE_PATH,
        candidate_path,
        device="cpu",
    )

    assert result.status == "failed"
    assert len(result.cases) == 1
    assert _passed_case_count(result) == 0
    assert result.cases[0].error is not None
    assert "intentional failure" in result.cases[0].error


def test_correctness_uses_reference_inputs_and_init_inputs(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self, bias):",
                "        super().__init__()",
                "        self.bias = bias",
                "",
                "    def forward(self, x):",
                "        return x + self.bias",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return [1.5]",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self, bias):",
                "        super().__init__()",
                "        self.bias = bias",
                "",
                "    def forward(self, x):",
                "        return x + self.bias",
                "",
                "def get_inputs():",
                "    return [torch.full((2, 2), 7.0)]",
                "",
                "def get_init_inputs():",
                "    return [9.0]",
            ]
        ),
    )
    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )

    # Input artifacts are now seed-only (no .pt files written).
    assert result.status == "passed"
    artifact = result.cases[0].input_artifact
    assert artifact is not None
    assert artifact["format"] == "manual_seed"
    assert isinstance(artifact["seed"], int) and artifact["seed"] >= 0
    assert "path" not in artifact, "tensor checkpoint should NOT be written"


def test_dict_output_passes_with_by_key_pairing(tmp_path: Path) -> None:
    """Reference and candidate both return dict[str, Tensor].

    Comparison must pair tensors by key, not by dict insertion order.
    """
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'quantized': x * 2.0, 'scales': x + 1.0}",
                "",
                "def get_inputs():",
                "    return [torch.ones(3, 3)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    # Candidate intentionally builds the dict in REVERSE insertion order
    # to prove pairing is by-key (alphabetical via flatten_outputs) and
    # not by Python dict iteration order.
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        scales = x + 1.0",
                "        quantized = x * 2.0",
                "        return {'scales': scales, 'quantized': quantized}",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )
    assert result.status == "passed", (
        f"unexpected failure: {result.reason or result.cases[0].error}"
    )
    assert result.cases[0].error is None
    output_names = sorted(diff.name for diff in result.cases[0].outputs)
    assert output_names == ["quantized", "scales"]
    for diff in result.cases[0].outputs:
        assert diff.passed is True
        assert diff.max_elementwise_abs_diff == 0.0


def test_dict_reference_vs_tuple_candidate_reports_structure_mismatch(tmp_path: Path) -> None:
    """When reference returns dict but candidate returns tuple, fail with a clear mismatch error."""
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'a': x * 2.0, 'b': x + 1.0}",
                "",
                "def get_inputs():",
                "    return [torch.ones(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate_tuple.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return (x * 2.0, x + 1.0)",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )
    assert result.status == "failed"
    assert _passed_case_count(result) == 0
    case_error = result.cases[0].error or ""
    assert "Output structure mismatch" in case_error
    assert "dict" in case_error
    assert "tuple" in case_error
    # Detailed structural diagnostics should mention the mismatched sides clearly
    assert "reference" in case_error.lower()
    assert "candidate" in case_error.lower()


def test_dict_outputs_with_mismatched_keys_reports_key_mismatch(tmp_path: Path) -> None:
    """When both sides return dict but with different keys, fail with a key-set diff."""
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'quantized': x * 2.0, 'scales': x + 1.0}",
                "",
                "def get_inputs():",
                "    return [torch.ones(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate_wrong_keys.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'q': x * 2.0, 's': x + 1.0}",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )
    assert result.status == "failed"
    case_error = result.cases[0].error or ""
    assert "Output structure mismatch" in case_error
    assert "dict keys differ" in case_error
    assert "missing in candidate" in case_error
    assert "extra in candidate" in case_error


# ---------------------------------------------------------------------------
# accuracy_mode=flashinfer: ported FlashInfer default_check semantics
# ---------------------------------------------------------------------------

import pytest  # noqa: E402  (appended section keeps imports local to the block)
import torch  # noqa: E402

from atrex_bench.eval.correctness import (  # noqa: E402
    ACCURACY_DTYPE_TOLERANCES_ENV,
    ACCURACY_MODE_ENV,
    ACCURACY_MODE_FLASHINFER,
    _cosine_similarity,
    flashinfer_default_tolerances,
)


def _write_pair(
    tmp_path: Path, *, ref_forward: str, cand_forward: str, inputs: str
) -> tuple[Path, Path]:
    """Write a reference/candidate pair sharing one get_inputs() definition."""
    header = ["import torch", "import torch.nn as nn", "", "class Model(nn.Module):"]
    tail = [
        "",
        "def get_inputs():",
        f"    return [{inputs}]",
        "",
        "def get_init_inputs():",
        "    return []",
    ]
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(header + ["    def forward(self, x):", f"        return {ref_forward}"] + tail),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(header + ["    def forward(self, x):", f"        return {cand_forward}"] + tail),
    )
    return reference_path, candidate_path


def test_flashinfer_default_tolerances_ladder() -> None:
    """The ported tier table matches flashinfer/trace/template.py exactly."""
    assert flashinfer_default_tolerances(torch.float64) == (1e-7, 1e-7)
    assert flashinfer_default_tolerances(torch.float32) == (1e-5, 1e-5)
    assert flashinfer_default_tolerances(torch.float16) == (1e-3, 1e-3)
    assert flashinfer_default_tolerances(torch.bfloat16) == (1e-2, 1e-2)
    assert flashinfer_default_tolerances(torch.float8_e4m3fn) == (1e-1, 1e-1)
    # Non-float dtypes fall through to exact equality.
    assert flashinfer_default_tolerances(torch.int64) == (0.0, 0.0)


def test_cosine_similarity_helper() -> None:
    ref = torch.ones(8, dtype=torch.float64)
    assert _cosine_similarity(ref, ref) == pytest.approx(1.0)
    assert _cosine_similarity(ref * 1.2, ref) == pytest.approx(1.0)  # scale-invariant
    assert _cosine_similarity(-ref, ref) == pytest.approx(-1.0)
    assert _cosine_similarity(torch.zeros(8, dtype=torch.float64), ref) == 0.0
    # Non-finite entries are filtered from both sides before the dot product.
    noisy = ref.clone()
    noisy[0] = float("nan")
    ref_noisy = ref.clone()
    ref_noisy[0] = float("inf")
    assert _cosine_similarity(noisy, ref_noisy) == pytest.approx(1.0)


def test_flashinfer_mode_exact_match_records_diagnostics(tmp_path: Path) -> None:
    reference_path, candidate_path = _write_pair(
        tmp_path, ref_forward="x + 1.0", cand_forward="x + 1.0", inputs="torch.zeros(4, 4)"
    )
    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
    )
    assert result.status == "passed"
    diff = result.cases[0].outputs[0]
    assert diff.passed is True
    assert diff.mismatch_pct == 0.0
    assert diff.cos_sim == pytest.approx(1.0)


def test_flashinfer_dtype_tolerances_are_stricter_than_global(tmp_path: Path) -> None:
    """A 1e-4 perturbation passes global atol=1e-2 but fails the fp32 tier (1e-5)."""
    reference_path, candidate_path = _write_pair(
        tmp_path, ref_forward="x + 1.0", cand_forward="x + 1.0001", inputs="torch.zeros(4, 4)"
    )
    strict = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_dtype_tolerances=True,
    )
    assert strict.status == "failed"
    assert strict.cases[0].outputs[0].mismatch_pct == pytest.approx(100.0)

    lenient = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_dtype_tolerances=False,  # explicit/global atol-rtol win
    )
    assert lenient.status == "passed"


def test_flashinfer_max_mismatch_pct_tolerates_outliers(tmp_path: Path) -> None:
    """10% of elements wrong: fails strict, passes with a 15% cap and cosine off."""
    reference_path, candidate_path = _write_pair(
        tmp_path,
        ref_forward="x",
        cand_forward="x * (torch.arange(x.numel(), dtype=x.dtype)"
        " .reshape(x.shape) >= x.numel() // 10)",
        inputs="torch.ones(10, 100)",
    )
    strict = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
    )
    assert strict.status == "failed"
    strict_diff = strict.cases[0].outputs[0]
    assert strict_diff.mismatch_pct == pytest.approx(10.0)
    # Cosine of 90%-ones vs all-ones is ~0.9487 < the 0.999 default floor, so
    # the mismatch cap alone must not flip the verdict.
    capped_only = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_max_mismatch_pct=15.0,
    )
    assert capped_only.status == "failed"
    assert capped_only.cases[0].outputs[0].cos_sim < 0.999

    passing = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_max_mismatch_pct=15.0,
        accuracy_min_cos_sim=-1.0,  # disables the cosine criterion
    )
    assert passing.status == "passed"


def test_flashinfer_cosine_only_gemm_convention(tmp_path: Path) -> None:
    """FlashInfer GEMM convention: mismatch cap 100% + cos floor 0.99."""
    reference_path, candidate_path = _write_pair(
        tmp_path, ref_forward="x", cand_forward="x * 1.2", inputs="torch.ones(10, 10)"
    )
    default = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
    )
    # Uniform 20% overshoot: every element fails isclose, cosine is exactly 1.
    assert default.status == "failed"
    assert default.cases[0].outputs[0].mismatch_pct == pytest.approx(100.0)
    assert default.cases[0].outputs[0].cos_sim == pytest.approx(1.0)

    gemm_style = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_max_mismatch_pct=100.0,
        accuracy_min_cos_sim=0.99,
    )
    assert gemm_style.status == "passed"


def test_flashinfer_mode_rejects_max_rel_l2_combination(tmp_path: Path) -> None:
    reference_path, candidate_path = _write_pair(
        tmp_path, ref_forward="x + 1.0", cand_forward="x + 1.0", inputs="torch.zeros(4, 4)"
    )
    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        max_rel_l2=0.1,
    )
    assert result.status == "failed"
    assert "cannot be combined" in (result.reason or "")


def test_flashinfer_mode_env_channel(tmp_path: Path, monkeypatch) -> None:
    """The ATREX_ACCURACY_* env vars (worker channel) drive the same engine."""
    reference_path, candidate_path = _write_pair(
        tmp_path, ref_forward="x + 1.0", cand_forward="x + 1.0001", inputs="torch.zeros(4, 4)"
    )
    monkeypatch.setenv(ACCURACY_MODE_ENV, "flashinfer")
    monkeypatch.setenv(ACCURACY_DTYPE_TOLERANCES_ENV, "1")
    result = check_correctness(reference_path, candidate_path, device="cpu")
    assert result.status == "failed"
    assert result.cases[0].outputs[0].mismatch_pct == pytest.approx(100.0)

    monkeypatch.delenv(ACCURACY_MODE_ENV)
    monkeypatch.delenv(ACCURACY_DTYPE_TOLERANCES_ENV)
    baseline = check_correctness(reference_path, candidate_path, device="cpu")
    assert baseline.status == "passed"
    assert baseline.cases[0].outputs[0].mismatch_pct is None


def test_allclose_mode_leaves_flashinfer_diagnostics_unset(tmp_path: Path) -> None:
    """Default mode payload stays exactly as before: no mismatch/cos fields."""
    result = check_correctness(REFERENCE_PATH, CANDIDATE_PATH, device="cpu")
    assert result.status == "passed"
    for diff in result.cases[0].outputs:
        assert diff.mismatch_pct is None
        assert diff.cos_sim is None


def test_flashinfer_max_mean_abs_err_criterion(tmp_path: Path) -> None:
    """SM120 NVFP4-attention style: MAE ceiling as an independent criterion."""
    reference_path, candidate_path = _write_pair(
        tmp_path, ref_forward="x", cand_forward="x * 1.2", inputs="torch.ones(10, 100)"
    )
    # Uniform 20% overshoot: cos == 1.0 and (with cap 100) mismatch passes,
    # so only the MAE ceiling can fail this candidate (mean |diff| = 0.2).
    failing = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_max_mismatch_pct=100.0,
        accuracy_min_cos_sim=0.99,
        accuracy_max_mean_abs_err=0.1,
    )
    assert failing.status == "failed"
    diff = failing.cases[0].outputs[0]
    assert diff.mean_abs_err == pytest.approx(0.2)

    passing = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
        accuracy_mode=ACCURACY_MODE_FLASHINFER,
        accuracy_max_mismatch_pct=100.0,
        accuracy_min_cos_sim=0.99,
        accuracy_max_mean_abs_err=0.3,
    )
    assert passing.status == "passed"


def test_mean_abs_err_unset_in_allclose_mode(tmp_path: Path) -> None:
    result = check_correctness(REFERENCE_PATH, CANDIDATE_PATH, device="cpu")
    assert result.status == "passed"
    assert result.cases[0].outputs[0].mean_abs_err is None
