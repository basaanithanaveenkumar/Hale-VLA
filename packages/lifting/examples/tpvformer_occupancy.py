"""TPVFormer for 3D semantic occupancy + LiDAR-style point queries.

python examples/tpvformer_occupancy.py --steps 200
"""

import argparse

import torch

from lifting import GridSpec, build_tpvformer
from lifting.data import SceneBatch, SyntheticSceneDataset
from lifting.training import OccupancyTask, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    grid = GridSpec(resolution=(32, 32, 4))
    data = SyntheticSceneDataset(256, grid, seed=1)
    batches = [SceneBatch.collate([data[i + j] for j in range(4)]) for i in range(0, 256, 4)]

    model = build_tpvformer(grid, num_classes=3, channels=32)
    trainer = Trainer(model, OccupancyTask(num_classes=3))
    trainer.fit(batches, args.steps)
    print("occupancy mIoU:", trainer.evaluate(batches[:8]).mean_foreground())

    # query arbitrary 3D points (e.g. a LiDAR sweep) through the tri-perspective planes
    batch = batches[0]
    points = torch.rand(4, 1000, 3) * torch.tensor([32.0, 32.0, 4.0]) - torch.tensor(
        [16.0, 16.0, 1.0]
    )
    with torch.no_grad():
        out = model.eval()(batch.images, batch.cameras, points)
    print("planes:", out["planes"].xy.shape, out["planes"].xz.shape, out["planes"].yz.shape)
    print("point logits:", out["point_logits"].shape)


if __name__ == "__main__":
    main()
