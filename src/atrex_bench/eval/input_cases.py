"""Opt-in, reproducible correctness input distributions.

Targets are explicit forward-argument paths, so data transformations never guess
which tensors are indices, scales, packed encodings, or rank contributions.
"""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass, field

import torch

from atrex_bench.eval._runtime import ModelInputs, deterministic_input_seed

PROFILE_DEFAULTS = {
    "wide_uniform": {"low": -16.0, "high": 16.0},
    "sparse_outliers": {"density": 0.01, "magnitude": 64.0},
    "log_uniform": {"min_exponent": -4.0, "max_exponent": 4.0},
    "cross_rank_cancellation": {"rank_dim": 0, "magnitude": 256.0, "residual": 1.0},
    "zeros": {},
    "tiny_values": {"multiplier": 1.0},
    "nonlinear_saturation": {"magnitude": 32.0},
    "routing_all_ties": {"value": 1.0},
    "rounding_near_ties": {"rounding_dtype": "bfloat16"},
    "signed_scale": {"magnitude": 1.0},
    "zero_scale": {},
    "fp4_extreme_codes": {},
}
INPUT_CONFIG_KEYS = ("correctness_seeds",) + tuple(
    f"correctness_{name}" for name in PROFILE_DEFAULTS
)
_ROUNDING_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class InputCase:
    seed: int
    profile: str | None = None
    parameters: dict = field(default_factory=dict)

    def artifact(self) -> dict:
        result = {"seed": self.seed, "format": "manual_seed"}
        if self.profile is not None:
            result.update(
                format="manual_seed_profile",
                profile=self.profile,
                profile_version=1,
                parameters=copy.deepcopy(self.parameters),
            )
        return result


@dataclass(frozen=True)
class CorrectnessInputConfig:
    seeds: tuple[int, ...] | None = None
    profiles: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_options(cls, options: dict) -> CorrectnessInputConfig | None:
        seeds = options.get("correctness_seeds")
        if seeds is not None:
            if (
                not isinstance(seeds, (list, tuple))
                or not seeds
                or any(type(seed) is not int or not 0 <= seed <= 2**31 - 1 for seed in seeds)
            ):
                raise ValueError("correctness_seeds must be a non-empty list of int32 seeds")
            if len(set(seeds)) != len(seeds):
                raise ValueError("correctness_seeds must not contain duplicates")
            seeds = tuple(seeds)
        profiles = {}
        for name, defaults in PROFILE_DEFAULTS.items():
            key = f"correctness_{name}"
            value = options.get(key)
            if value is None or value is False:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object with inputs, or false/null to disable")
            unknown = value.keys() - defaults.keys() - {"inputs"}
            if unknown:
                raise ValueError(f"{key}: unknown parameter(s): {sorted(unknown)}")
            targets = value.get("inputs")
            if (
                not isinstance(targets, list)
                or not targets
                or any(
                    not isinstance(path, str)
                    or not path
                    or any(not part for part in path.split("."))
                    for path in targets
                )
            ):
                raise ValueError(f"{key}.inputs must be a non-empty list of argument paths")
            if len(set(targets)) != len(targets):
                raise ValueError(f"{key}.inputs must not contain duplicates")
            params = {**defaults, **copy.deepcopy(value)}
            for param, val in params.items():
                if param == "inputs":
                    continue
                if param == "rounding_dtype":
                    if not isinstance(val, str) or val not in _ROUNDING_DTYPES:
                        raise ValueError(f"{key}.rounding_dtype must be float16 or bfloat16")
                elif param == "rank_dim":
                    if type(val) is not int:
                        raise ValueError(f"{key}.rank_dim must be an integer")
                elif type(val) not in (int, float) or not math.isfinite(val):
                    raise ValueError(f"{key}.{param} must be finite numeric data")
            if name == "wide_uniform" and params["low"] >= params["high"]:
                raise ValueError(f"{key} requires low < high")
            if name == "log_uniform" and not (
                -37 <= params["min_exponent"] < params["max_exponent"] <= 37
            ):
                raise ValueError(f"{key} requires -37 <= min_exponent < max_exponent <= 37")
            if name == "sparse_outliers" and not 0 < params["density"] < 1:
                raise ValueError(f"{key}.density must be between 0 and 1 (exclusive)")
            for param in ("magnitude", "multiplier"):
                if param in params and params[param] <= 0:
                    raise ValueError(f"{key}.{param} must be positive")
            profiles[name] = params
        return cls(seeds, profiles) if seeds is not None or profiles else None

    def to_options(self) -> dict:
        options = {f"correctness_{name}": copy.deepcopy(p) for name, p in self.profiles.items()}
        if self.seeds is not None:
            options["correctness_seeds"] = list(self.seeds)
        return options

    def cli_args(self) -> list[str]:
        import json

        return [
            item
            for name, value in self.to_options().items()
            for item in ("--" + name.replace("_", "-"), json.dumps(value))
        ]

    def case_count(self, num_cases: int) -> int:
        return (
            num_cases
            * (len(self.seeds) if self.seeds is not None else 1)
            * (1 + len(self.profiles))
        )

    def cases(self, shape_id: str, num_cases: int) -> list[InputCase]:
        if self.seeds is None:
            seeds = [deterministic_input_seed("correctness", shape_id, i) for i in range(num_cases)]
        else:
            # One draw uses the supplied seed verbatim. Extra draws have stable,
            # separately recorded seeds and do not depend on enabled profiles.
            seeds = [
                seed if i == 0 else deterministic_input_seed("correctness_draw", str(seed), i)
                for seed in self.seeds
                for i in range(num_cases)
            ]
        return [
            InputCase(seed, profile, copy.deepcopy(params))
            for seed in seeds
            for profile, params in [(None, {}), *self.profiles.items()]
        ]


def _target(arguments: dict, path: str):
    value = arguments
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError(f"Unknown correctness input target: {path}")
    return value


def _float_tensor(value, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{path} must target a floating-point tensor")
    if value.numel() == 0:
        raise ValueError(f"{path} targets an empty tensor; distribution cannot be exercised")
    return value


def _copy_values(target: torch.Tensor, values: torch.Tensor) -> None:
    # Copy through the original view to preserve layout, strides and aliasing.
    converted = values.to(dtype=target.dtype, device=target.device)
    if converted.is_floating_point() and not torch.isfinite(converted.float()).all().item():
        raise ValueError("Input profile overflows target dtype; reduce its magnitude/range")
    target.copy_(converted)


def _pattern(tensor: torch.Tensor, values: list[float]) -> torch.Tensor:
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    palette = torch.tensor(values, dtype=dtype, device=tensor.device)
    return palette.repeat(math.ceil(tensor.numel() / len(values)))[: tensor.numel()].reshape(
        tensor.shape
    )


def _cancel_ranks(value, path: str, params: dict) -> None:
    if isinstance(value, (list, tuple)):
        ranks = [_float_tensor(item, f"{path}.{i}") for i, item in enumerate(value)]
    else:
        tensor = _float_tensor(value, path)
        dim = params["rank_dim"]
        if not -tensor.ndim <= dim < tensor.ndim:
            raise ValueError(f"{path}: rank_dim is outside tensor dimensions")
        ranks = list(tensor.unbind(dim))
    if len(ranks) < 2 or any(
        r.shape != ranks[0].shape or r.dtype != ranks[0].dtype or r.device != ranks[0].device
        for r in ranks
    ):
        raise ValueError(f"{path}: cancellation requires at least two matching rank contributions")
    for i in range(0, len(ranks) - 1, 2):
        base = torch.empty(ranks[i].shape, device=ranks[i].device).uniform_(0.5, 1.0)
        _copy_values(ranks[i], base * params["magnitude"])
        _copy_values(ranks[i + 1], -ranks[i].float() + params["residual"])
    if len(ranks) % 2:
        _copy_values(ranks[-1], torch.full_like(ranks[-1], params["residual"]))


def apply_input_case(inputs: ModelInputs, signature: inspect.Signature, case: InputCase) -> None:
    """Apply one profile after seeding RNGs and calling the original input factory.

    This is also the replay entry point for ``manual_seed_profile`` artifacts.
    A baseline case is a no-op. The caller owns these fresh input tensors.
    """
    if case.profile is None:
        return
    arguments = signature.bind(*inputs.args, **inputs.kwargs).arguments
    name, params = case.profile, case.parameters
    with torch.no_grad():
        for path in params["inputs"]:
            value = _target(arguments, path)
            if name == "cross_rank_cancellation":
                _cancel_ranks(value, path, params)
                continue
            if name == "fp4_extreme_codes":
                if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
                    raise ValueError(f"{path}: FP4 E2M1 packed data must be a uint8 tensor")
                if value.numel() == 0:
                    raise ValueError(f"{path}: cannot exercise FP4 codes on an empty tensor")
                # E2M1 nibbles: +0, -0, +6, -6. Exercise both nibble positions
                # and all 16 combinations, including opposite signs in one byte.
                codes = [0x0, 0x8, 0x7, 0xF]
                packed_codes = [(hi << 4) | lo for hi in codes for lo in codes]
                _copy_values(value, _pattern(value, packed_codes))
                continue
            tensor = _float_tensor(value, path)
            if name == "wide_uniform":
                values = torch.empty(tensor.shape, device=tensor.device).uniform_(
                    params["low"], params["high"]
                )
            elif name == "sparse_outliers":
                values = torch.zeros(tensor.numel(), device=tensor.device)
                count = max(1, math.ceil(tensor.numel() * params["density"]))
                indices = torch.randperm(tensor.numel(), device=tensor.device)[:count]
                signs = torch.randint(0, 2, (count,), device=tensor.device) * 2 - 1
                values[indices] = signs.float() * params["magnitude"]
                values = values.reshape(tensor.shape)
            elif name == "log_uniform":
                exponents = torch.empty(tensor.shape, device=tensor.device).uniform_(
                    params["min_exponent"], params["max_exponent"]
                )
                signs = torch.randint(0, 2, tensor.shape, device=tensor.device) * 2 - 1
                values = 10.0**exponents * signs
            elif name in {"zeros", "zero_scale"}:
                values = torch.zeros(tensor.shape, device=tensor.device)
            elif name == "tiny_values":
                tiny = torch.finfo(tensor.dtype).tiny * params["multiplier"]
                values = _pattern(tensor, [tiny, -tiny, tiny / 2, -tiny / 2])
            elif name in {"nonlinear_saturation", "signed_scale"}:
                magnitude = params["magnitude"]
                signs = (
                    [-magnitude, magnitude] if name == "signed_scale" else [magnitude, -magnitude]
                )
                values = _pattern(tensor, signs)
            elif name == "routing_all_ties":
                if tensor.ndim == 0 or tensor.shape[-1] < 2:
                    raise ValueError(
                        f"{path}: routing requires at least two entries on the last axis"
                    )
                values = torch.full(tensor.shape, params["value"], device=tensor.device)
            elif name == "rounding_near_ties":
                if tensor.ndim == 0 or tensor.shape[-1] < 4:
                    raise ValueError(
                        f"{path}: near ties requires at least four entries on the last axis"
                    )
                dtype = _ROUNDING_DTYPES[params["rounding_dtype"]]
                eps = torch.finfo(dtype).eps
                # Perturbations collapse into tied groups on adjacent rounded
                # levels. Restart each row so every routing row sees both groups.
                row_tensor = torch.empty(tensor.shape[-1], device=tensor.device)
                row = _pattern(
                    row_tensor, [1 - eps / 4, 1 + eps / 4, 1 + 3 * eps / 4, 1 + 5 * eps / 4]
                )
                values = row.to(dtype).float().expand(tensor.shape)
                if torch.finfo(tensor.dtype).eps > eps:
                    raise ValueError(f"{path}: target dtype is coarser than rounding_dtype")
            else:
                raise ValueError(f"Unknown correctness profile: {name}")
            _copy_values(tensor, values)
