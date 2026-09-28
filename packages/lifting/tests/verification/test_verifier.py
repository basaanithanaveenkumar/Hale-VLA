"""The verification pipeline: judging logic, corruption, reports, GIF and CLI."""

import json

import pytest
import torch
from PIL import Image

from lifting.cli import main
from lifting.geometry import GridSpec
from lifting.verification import (
    ExtrinsicCorruption,
    LifterReport,
    LiftingVerifier,
    VerificationConfig,
    VerificationGifRenderer,
    VerificationReport,
)

TINY = dict(
    steps=3,
    batch_size=2,
    train_scenes=4,
    val_scenes=4,
    image_size=(32, 64),
    grid=GridSpec(resolution=(16, 16, 4)),
    channels=16,
    snapshots=2,
)


def report(name="x", task="bev", miou=0.5, corrupted=0.05, loss=(1.0, 0.2)) -> LifterReport:
    return LifterReport(name, task, True, [0.9, miou, miou], miou, corrupted, loss[0], loss[1], 1.0)


def test_judge_accepts_a_good_lifter() -> None:
    v = LiftingVerifier(VerificationConfig(), log=None)
    r = report()
    v.judge(r, report("baseline", miou=0.1, corrupted=0.1))
    assert r.passed and set(r.checks) == {"beats_baseline", "uses_geometry", "loss_decreases"}


@pytest.mark.parametrize(
    "kwargs,failing",
    [
        (dict(miou=0.2), "beats_baseline"),  # not better than the baseline
        (dict(corrupted=0.45), "uses_geometry"),  # ignores calibration
        (dict(loss=(1.0, 0.9)), "loss_decreases"),  # did not learn
    ],
)
def test_judge_rejects_bad_lifters(kwargs, failing) -> None:
    v = LiftingVerifier(VerificationConfig(), log=None)
    r = report(**kwargs)
    v.judge(r, report("baseline", miou=0.1, corrupted=0.1))
    assert not r.passed and not r.checks[failing]


def test_occupancy_judging() -> None:
    v = LiftingVerifier(VerificationConfig(min_occupancy_miou=0.3), log=None)
    r = report(task="occupancy", miou=0.25, corrupted=0.01)
    v.judge(r, report())
    assert r.checks["occupancy_miou"] is False and r.checks["uses_geometry"]


def test_extrinsic_corruption(batch) -> None:
    corrupted = ExtrinsicCorruption(90.0)(batch)
    assert torch.equal(corrupted.images, batch.images)
    assert torch.equal(corrupted.bev_labels, batch.bev_labels)
    assert not torch.allclose(corrupted.cameras.cam_to_ego, batch.cameras.cam_to_ego)
    assert torch.allclose(
        corrupted.cameras.cam_to_ego[..., 2, 3], batch.cameras.cam_to_ego[..., 2, 3]
    )


def test_tiny_run_produces_report_and_gif(tmp_path) -> None:
    verifier = LiftingVerifier(VerificationConfig(lifters=("simple_bev",), **TINY), log=None)
    rep = verifier.run()
    assert isinstance(rep, VerificationReport)
    assert [r.name for r in rep.lifters] == ["simple_bev", "tpvformer/occupancy"]
    assert rep.baseline.name == "mlp_view"
    assert all(len(r.snapshots) == 2 for r in [rep.baseline, *rep.lifters])
    data = json.loads(rep.to_json())
    assert set(data) == {"passed", "baseline", "lifters"}
    assert "simple_bev" in rep.summary()

    path = VerificationGifRenderer().save(rep, verifier.visualisation_batch, tmp_path / "v.gif")
    gif = Image.open(path)
    assert gif.n_frames == 2 + 2  # snapshots + final + corrupted


def test_cli_writes_json(tmp_path) -> None:
    out = tmp_path / "r.json"
    code = main(
        [
            "--lifters",
            "simple_bev",
            "--steps",
            "2",
            "--train-scenes",
            "4",
            "--val-scenes",
            "4",
            "--channels",
            "16",
            "--no-occupancy",
            "--gif",
            str(tmp_path / "v.gif"),
            "--json",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert code == (0 if data["passed"] else 1)
    assert (tmp_path / "v.gif").exists() == data["passed"]


@pytest.mark.slow
def test_full_verification_passes(tmp_path) -> None:
    """The real thing: every lifter learns from geometry on synthetic 3D scenes (minutes on CPU)."""
    verifier = LiftingVerifier(VerificationConfig(), log=print)
    rep = verifier.run()
    assert rep.passed, rep.summary()
    VerificationGifRenderer().save(rep, verifier.visualisation_batch, tmp_path / "verification.gif")
