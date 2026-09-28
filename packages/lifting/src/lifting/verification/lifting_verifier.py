"""End-to-end verification that every lifter really lifts.

For each lifter the verifier trains a full model on procedurally rendered 3D scenes
(:class:`SyntheticSceneDataset`) and checks three properties:

1. **beats_baseline** - its BEV segmentation mIoU exceeds a geometry-free control
   (:class:`MLPViewLifter`) by a margin. Because the camera rig is randomly rotated in
   every scene, no model can succeed without using the calibration.
2. **uses_geometry** - with deliberately rotated extrinsics (images and labels unchanged)
   its mIoU collapses, proving predictions are placed through camera geometry.
3. **loss_decreases** - the training loss drops substantially.

TPVFormer is additionally trained for 3D semantic occupancy.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from lifting.data import CameraRig, SceneBatch, SyntheticSceneDataset
from lifting.models import build_bev_model, build_tpvformer
from lifting.training import BEVSegmentationTask, OccupancyTask, Task, Trainer
from lifting.verification.extrinsic_corruption import ExtrinsicCorruption
from lifting.verification.lifter_report import LifterReport
from lifting.verification.verification_config import VerificationConfig
from lifting.verification.verification_report import VerificationReport

NUM_CLASSES = 3


class LiftingVerifier:
    """Trains and checks lifters on synthetic scenes (see module docstring)."""

    def __init__(
        self, config: VerificationConfig | None = None, log: Callable[[str], None] | None = print
    ) -> None:
        self.config = config or VerificationConfig()
        self.log = log or (lambda _msg: None)
        self.corruption = ExtrinsicCorruption(self.config.corruption_yaw_deg)

    # ------------------------------------------------------------------ data
    def build_data(self) -> tuple[list[SceneBatch], list[SceneBatch]]:
        """Pre-render training and validation batches (deterministic)."""
        cfg = self.config
        rig = CameraRig(cfg.num_cameras, cfg.image_size, yaw_jitter_deg=cfg.rig_yaw_jitter_deg)
        train = SyntheticSceneDataset(cfg.train_scenes, cfg.grid, rig, seed=cfg.seed + 1)
        val = SyntheticSceneDataset(cfg.val_scenes, cfg.grid, rig, seed=cfg.seed + 2)
        return self._batches(train, cfg.batch_size), self._batches(val, 2 * cfg.batch_size)

    @staticmethod
    def _batches(dataset: SyntheticSceneDataset, size: int) -> list[SceneBatch]:
        return [
            SceneBatch.collate([dataset[i] for i in range(j, min(j + size, len(dataset)))])
            for j in range(0, len(dataset), size)
        ]

    # ------------------------------------------------------------------ training
    def _snapshot_steps(self) -> set[int]:
        s, n = self.config.snapshots, self.config.steps
        if s <= 1:
            return {n - 1}
        return {round(i * (n - 1) / (s - 1)) for i in range(s)}

    def _train_and_evaluate(
        self,
        name: str,
        task_name: str,
        model: torch.nn.Module,
        task: Task,
        uses_geometry: bool,
        train: list[SceneBatch],
        val: list[SceneBatch],
    ) -> LifterReport:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        trainer = Trainer(model, task, lr=cfg.lr, device=cfg.device)
        vis = val[0]
        snapshots: list[tuple[int, torch.Tensor]] = []
        wanted = self._snapshot_steps()

        def record(step: int, _loss: float) -> None:
            if step in wanted:
                snapshots.append((step + 1, trainer.predict(vis)[0].cpu()))

        start = time.perf_counter()
        history = trainer.fit(train, cfg.steps, callbacks=[record])
        seconds = time.perf_counter() - start
        clean = trainer.evaluate(val)
        corrupted = trainer.evaluate(val, transform=self.corruption)
        smooth = history.smoothed(20)
        report = LifterReport(
            name=name,
            task=task_name,
            uses_geometry=uses_geometry,
            iou_per_class=[round(float(v), 4) for v in clean.per_class()],
            miou=clean.mean_foreground(),
            miou_corrupted=corrupted.mean_foreground(),
            initial_loss=sum(history.losses[:10]) / max(1, len(history.losses[:10])),
            final_loss=smooth[-1],
            seconds=seconds,
            snapshots=snapshots,
            corrupted_prediction=trainer.predict(self.corruption(vis.to(cfg.device)))[0].cpu(),
        )
        self.log(
            f"[{name}] mIoU={report.miou:.3f} corrupted={report.miou_corrupted:.3f} "
            f"loss {report.initial_loss:.3f}->{report.final_loss:.3f} ({seconds:.0f}s)"
        )
        return report

    def verify_bev(
        self, lifter: str, train: list[SceneBatch], val: list[SceneBatch]
    ) -> LifterReport:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        model = build_bev_model(
            lifter, cfg.grid, NUM_CLASSES, cfg.image_size, cfg.num_cameras, cfg.channels
        )
        return self._train_and_evaluate(
            lifter,
            "bev",
            model,
            BEVSegmentationTask(NUM_CLASSES),
            model.lifter.uses_geometry,
            train,
            val,
        )

    def verify_occupancy(self, train: list[SceneBatch], val: list[SceneBatch]) -> LifterReport:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        model = build_tpvformer(cfg.grid, NUM_CLASSES, cfg.channels)
        return self._train_and_evaluate(
            "tpvformer/occupancy", "occupancy", model, OccupancyTask(NUM_CLASSES), True, train, val
        )

    # ------------------------------------------------------------------ judging
    def judge(self, report: LifterReport, baseline: LifterReport) -> None:
        cfg = self.config
        if report.task == "bev":
            report.checks["beats_baseline"] = (
                report.miou >= baseline.miou + cfg.min_margin_over_baseline
            )
        else:
            report.checks["occupancy_miou"] = report.miou >= cfg.min_occupancy_miou
        report.checks["uses_geometry"] = (
            report.miou_corrupted <= (1 - cfg.min_relative_drop) * report.miou
        )
        report.checks["loss_decreases"] = (
            report.final_loss <= (1 - cfg.min_loss_reduction) * report.initial_loss
        )

    # ------------------------------------------------------------------ entry point
    def run(self) -> VerificationReport:
        self.log("rendering synthetic scenes ...")
        train, val = self.build_data()
        baseline = self.verify_bev(self.config.baseline, train, val)
        report = VerificationReport(baseline=baseline)
        for lifter in self.config.lifters:
            r = self.verify_bev(lifter, train, val)
            self.judge(r, baseline)
            report.lifters.append(r)
        if self.config.verify_occupancy:
            r = self.verify_occupancy(train, val)
            self.judge(r, baseline)
            report.lifters.append(r)
        self._vis_batch = val[0]
        self.log(report.summary())
        return report

    @property
    def visualisation_batch(self) -> SceneBatch:
        """The validation batch whose first scene is shown in snapshots (after :meth:`run`)."""
        return self._vis_batch
