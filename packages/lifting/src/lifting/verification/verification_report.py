"""Aggregated verification outcome."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from lifting.verification.lifter_report import LifterReport


@dataclass
class VerificationReport:
    """All lifter reports plus the baseline they were compared against."""

    baseline: LifterReport
    lifters: list[LifterReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.lifters)

    def summary(self) -> str:
        """Markdown table of the results."""
        lines = [
            "| lifter | task | mIoU | mIoU (corrupted extrinsics) | loss start -> end | time | checks |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in [self.baseline, *self.lifters]:
            checks = (
                ", ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in r.checks.items())
                or "baseline"
            )
            lines.append(
                f"| {r.name} | {r.task} | {r.miou:.3f} | {r.miou_corrupted:.3f} | "
                f"{r.initial_loss:.3f} -> {r.final_loss:.3f} | {r.seconds:.0f}s | {checks} |"
            )
        lines.append("")
        lines.append(f"**overall: {'PASS' if self.passed else 'FAIL'}**")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "baseline": self.baseline.to_dict(),
                "lifters": [r.to_dict() for r in self.lifters],
            },
            indent=2,
        )
