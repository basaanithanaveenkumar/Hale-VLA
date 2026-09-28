"""Adding a new lifter takes one class - nothing in the library has to change (open/closed).

python examples/custom_lifter.py
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from lifting import LIFTERS, BaseLifter, Cameras, GridSpec, build_bev_model
from lifting.data import SceneBatch, SyntheticSceneDataset
from lifting.layers import ConvNormAct
from lifting.ops import masked_camera_mean, sample_multi_camera


@LIFTERS.register("ground_plane")
class GroundPlaneLifter(BaseLifter):
    """Inverse perspective mapping: sample every BEV cell's *ground* point (z = 0) only."""

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.out = ConvNormAct(in_channels, out_channels)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        feats = features[0]
        b, _, c = feats.shape[:3]
        ground = self.grid.voxel_centers(feats.device)[:, :, 0].clone()
        ground[..., 2] = 0.0
        uv, _, valid = cameras.project(ground.view(1, -1, 3).expand(b, -1, -1))
        bev = masked_camera_mean(sample_multi_camera(feats, cameras.normalize_uv(uv)), valid)
        return self.out(bev.view(b, c, *self.grid.bev_shape))


if __name__ == "__main__":
    grid = GridSpec(resolution=(32, 32, 4))
    batch = SceneBatch.collate([SyntheticSceneDataset(2, grid)[i] for i in range(2)])
    model = build_bev_model("ground_plane", grid, num_classes=3, channels=32)
    with torch.no_grad():
        print(model(batch.images, batch.cameras)["logits"].shape)
