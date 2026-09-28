"""Every lifter must be trainable: the loss on a fixed batch has to go down."""

import pytest
import torch

from lifting import LIFTERS, GridSpec, build_bev_model, build_tpvformer
from lifting.data import SceneBatch
from lifting.training import BEVSegmentationTask, OccupancyTask, Trainer

IMAGE = (32, 64)


@pytest.mark.parametrize("name", LIFTERS.names())
def test_overfits_a_fixed_batch(name: str, batch: SceneBatch, grid: GridSpec) -> None:
    torch.manual_seed(0)
    model = build_bev_model(name, grid, 3, IMAGE, channels=16)
    trainer = Trainer(model, BEVSegmentationTask(3), lr=3e-3)
    history = trainer.fit([batch], steps=40)
    first, last = history.losses[0], sum(history.losses[-5:]) / 5
    assert last < 0.5 * first, f"{name}: loss {first:.3f} -> {last:.3f}"


def test_tpvformer_occupancy_training(batch: SceneBatch, grid: GridSpec) -> None:
    torch.manual_seed(0)
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1)
    trainer = Trainer(model, OccupancyTask(3), lr=3e-3)
    history = trainer.fit([batch], steps=40)
    assert sum(history.losses[-5:]) / 5 < 0.5 * history.losses[0]
    metric = trainer.evaluate([batch])
    assert metric.mean_foreground() > 0.2


def test_trainer_callbacks_evaluate_and_predict(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_bev_model("simple_bev", grid, 3, IMAGE, channels=16)
    trainer = Trainer(model, BEVSegmentationTask(3))
    seen: list[int] = []
    trainer.fit([batch, batch], steps=5, callbacks=[lambda step, loss: seen.append(step)])
    assert seen == [0, 1, 2, 3, 4]
    assert len(trainer.history.losses) == 5
    assert len(trainer.history.smoothed(3)) == 5
    assert trainer.predict(batch).shape == (2, *grid.bev_shape)
    metric = trainer.evaluate([batch])
    assert metric.confusion.sum() == batch.bev_labels.numel()


def test_trainer_rejects_empty_data(grid: GridSpec) -> None:
    trainer = Trainer(
        build_bev_model("simple_bev", grid, 3, IMAGE, channels=16), BEVSegmentationTask(3)
    )
    with pytest.raises(ValueError):
        trainer.fit([], steps=1)
