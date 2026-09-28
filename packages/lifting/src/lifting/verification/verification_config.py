"""Settings of a verification run."""

from __future__ import annotations

from dataclasses import dataclass, field

from lifting.geometry import GridSpec

GEOMETRIC_LIFTERS = ("simple_bev", "depth_warp", "lift_splat", "tiim", "bevformer", "tpvformer")


@dataclass
class VerificationConfig:
    """What to verify and how strict to be.

    Attributes:
        lifters: geometric lifters under test.
        baseline: geometry-free control lifter every method must beat.
        steps: optimisation steps per lifter.
        batch_size: training batch size.
        train_scenes, val_scenes: sizes of the synthetic splits.
        image_size, num_cameras: camera rig.
        grid: BEV / voxel grid.
        channels: model width.
        lr: learning rate.
        rig_yaw_jitter_deg: random rotation of the whole rig per scene - forces the
            network to read the extrinsics instead of memorising a fixed layout.
        corruption_yaw_deg: extrinsics are rotated by this much for the sensitivity test.
        min_margin_over_baseline: required BEV mIoU gain over the baseline.
        min_relative_drop: required relative mIoU drop under corrupted extrinsics.
        min_loss_reduction: required relative reduction of the smoothed training loss.
        verify_occupancy: also train TPVFormer on 3D semantic occupancy.
        min_occupancy_miou: required 3D occupancy mIoU for TPVFormer.
        snapshots: number of prediction snapshots recorded (GIF frames).
        seed: global seed.
        device: torch device.
    """

    lifters: tuple[str, ...] = GEOMETRIC_LIFTERS
    baseline: str = "mlp_view"
    steps: int = 300
    batch_size: int = 4
    train_scenes: int = 512
    val_scenes: int = 64
    image_size: tuple[int, int] = (48, 96)
    num_cameras: int = 4
    grid: GridSpec = field(default_factory=GridSpec)
    channels: int = 32
    lr: float = 2e-3
    rig_yaw_jitter_deg: float = 180.0
    corruption_yaw_deg: float = 90.0
    min_margin_over_baseline: float = 0.15
    min_relative_drop: float = 0.5
    min_loss_reduction: float = 0.3
    verify_occupancy: bool = True
    min_occupancy_miou: float = 0.2
    snapshots: int = 12
    seed: int = 0
    device: str = "cpu"
