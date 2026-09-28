from __future__ import annotations

import pytest
import torch

from lifting.data import CameraRig, SceneBatch, SyntheticSceneDataset
from lifting.geometry import Cameras, GridSpec

torch.set_num_threads(min(4, torch.get_num_threads()))


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(resolution=(16, 16, 4))


@pytest.fixture
def rig() -> CameraRig:
    return CameraRig(num_cameras=4, image_size=(32, 64), yaw_jitter_deg=180.0)


@pytest.fixture
def batch(grid: GridSpec, rig: CameraRig) -> SceneBatch:
    ds = SyntheticSceneDataset(2, grid=grid, rig=rig, seed=3)
    return SceneBatch.collate([ds[0], ds[1]])


@pytest.fixture
def cameras(batch: SceneBatch) -> Cameras:
    return batch.cameras
