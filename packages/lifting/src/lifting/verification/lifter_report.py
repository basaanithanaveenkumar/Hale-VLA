"""Outcome of verifying one lifter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from torch import Tensor


@dataclass
class LifterReport:
    """Metrics, pass/fail checks and prediction snapshots of one trained lifter.

    Attributes:
        name: lifter (or ``"<lifter>/occupancy"`` for the 3D task).
        task: ``"bev"`` or ``"occupancy"``.
        uses_geometry: whether the lifter reads camera calibration.
        iou_per_class: validation IoU per class (background first).
        miou: mean foreground IoU on clean validation data.
        miou_corrupted: mean foreground IoU with corrupted extrinsics.
        initial_loss, final_loss: smoothed training loss at the start / end.
        seconds: training time.
        checks: name -> passed.
        snapshots: ``(step, prediction)`` pairs on the visualisation scene.
        corrupted_prediction: prediction on the visualisation scene with corrupted extrinsics.
    """

    name: str
    task: str
    uses_geometry: bool
    iou_per_class: list[float]
    miou: float
    miou_corrupted: float
    initial_loss: float
    final_loss: float
    seconds: float
    checks: dict[str, bool] = field(default_factory=dict)
    snapshots: list[tuple[int, Tensor]] = field(default_factory=list, repr=False)
    corrupted_prediction: Tensor | None = field(default=None, repr=False)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("snapshots")
        data.pop("corrupted_prediction")
        data["passed"] = self.passed
        return data
