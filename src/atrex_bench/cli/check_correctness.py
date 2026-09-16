"""CLI for Stage 1: compare eager(reference) against a candidate module."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from atrex_bench.eval.correctness import check_correctness
from atrex_bench.utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check correctness between eager(reference) and a candidate module"
    )
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference.py")
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to the candidate Python file",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2,
        help="Absolute tolerance for near-zero outputs",
    )
    parser.add_argument("--rtol", type=float, default=5e-2, help="Relative tolerance")
    parser.add_argument(
        "--num-correctness-cases",
        type=int,
        default=1,
        help="Number of times to sample reference inputs via _make_inputs(**shape.input_kwargs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:0, hip, hip:0 ...",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Relative or absolute directory for correctness input checkpoints",
    )
    parser.add_argument(
        "--correctness-error-budgets", type=json.loads, default=None,
        help="Per-output JSON tolerances and optional RMS budgets (rms_atol, rms_rtol, dim).",
    )
    args = parser.parse_args()
    artifact_dir = None
    artifact_root = None
    if args.output is not None:
        artifact_root = args.output.parent
        if args.checkpoint_dir is None:
            artifact_dir = artifact_root / "correctness"
        elif args.checkpoint_dir.is_absolute():
            artifact_dir = args.checkpoint_dir
        else:
            artifact_dir = artifact_root / args.checkpoint_dir

    result = asdict(
        check_correctness(
            reference_path=args.reference,
            candidate_path=args.candidate,
            atol=args.atol,
            rtol=args.rtol,
            correctness_error_budgets=args.correctness_error_budgets,
            num_correctness_cases=args.num_correctness_cases,
            device=args.device,
            artifact_dir=artifact_dir,
            artifact_root=artifact_root,
        )
    )
    if args.output is not None:
        save_json(result, args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
