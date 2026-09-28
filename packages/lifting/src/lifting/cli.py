"""``lifting-verify`` command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lifting.verification import (
    GEOMETRIC_LIFTERS,
    LiftingVerifier,
    VerificationConfig,
    VerificationGifRenderer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify lifting techniques on synthetic 3D scenes.")
    p.add_argument(
        "--lifters", default=",".join(GEOMETRIC_LIFTERS), help="comma separated lifter names"
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-scenes", type=int, default=512)
    p.add_argument("--val-scenes", type=int, default=64)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument(
        "--no-occupancy", action="store_true", help="skip the TPVFormer 3D occupancy check"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gif", type=Path, default=Path("verification.gif"))
    p.add_argument("--json", type=Path, default=None, help="optional path for a JSON report")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = VerificationConfig(
        lifters=tuple(n for n in args.lifters.split(",") if n),
        steps=args.steps,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        channels=args.channels,
        verify_occupancy=not args.no_occupancy,
        seed=args.seed,
        device=args.device,
    )
    verifier = LiftingVerifier(config)
    report = verifier.run()
    if report.passed:
        path = VerificationGifRenderer().save(report, verifier.visualisation_batch, args.gif)
        print(f"verification passed - animation written to {path}")
    else:
        print("verification FAILED - no GIF written")
    if args.json:
        args.json.write_text(report.to_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
