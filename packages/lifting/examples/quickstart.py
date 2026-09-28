"""Train a BEV segmentation model on synthetic scenes with any lifter, in ~20 lines.

python examples/quickstart.py --lifter bevformer --steps 200
"""

import argparse

from lifting import LIFTERS, GridSpec, build_bev_model
from lifting.data import SceneBatch, SyntheticSceneDataset
from lifting.training import BEVSegmentationTask, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifter", default="simple_bev", choices=LIFTERS.names())
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    grid = GridSpec(resolution=(32, 32, 4))
    train = SyntheticSceneDataset(256, grid, seed=1)
    val = SyntheticSceneDataset(32, grid, seed=2)
    train_batches = [SceneBatch.collate([train[i + j] for j in range(4)]) for i in range(0, 256, 4)]
    val_batches = [SceneBatch.collate([val[i + j] for j in range(8)]) for i in range(0, 32, 8)]

    model = build_bev_model(args.lifter, grid, num_classes=3, channels=32)
    trainer = Trainer(model, BEVSegmentationTask(num_classes=3))
    trainer.fit(train_batches, args.steps, callbacks=[lambda s, loss: s % 50 or print(s, loss)])
    metric = trainer.evaluate(val_batches)
    print(
        f"{args.lifter}: per-class IoU {metric.per_class().tolist()}  mIoU {metric.mean_foreground():.3f}"
    )


if __name__ == "__main__":
    main()
