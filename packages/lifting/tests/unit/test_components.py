import pytest
import torch

from lifting.decoders import BEVDecoder, SegmentationHead
from lifting.encoders import ResNetEncoder, SimpleConvEncoder
from lifting.lifters import LIFTERS
from lifting.metrics import IoUMetric
from lifting.registry import Registry


def test_registry() -> None:
    reg: Registry[object] = Registry("thing")

    @reg.register("a")
    class A:
        def __init__(self, x: int = 1) -> None:
            self.x = x

    assert "a" in reg and reg.names() == ["a"]
    assert reg.build("a", x=3).x == 3
    with pytest.raises(KeyError):
        reg.get("missing")
    with pytest.raises(KeyError):
        reg.register("a")(A)


def test_all_lifters_registered() -> None:
    assert set(LIFTERS.names()) == {
        "simple_bev",
        "depth_warp",
        "lift_splat",
        "bevformer",
        "tiim",
        "tpvformer",
        "mlp_view",
    }


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_simple_encoder_strides(levels: int) -> None:
    enc = SimpleConvEncoder(out_channels=16, num_levels=levels)
    feats = enc(torch.rand(2, 3, 32, 64))
    assert len(feats) == levels == enc.num_levels
    for f, s in zip(feats, enc.strides, strict=True):
        assert f.shape == (2, 16, 32 // s, 64 // s)


def test_resnet_encoder() -> None:
    pytest.importorskip("torchvision")
    enc = ResNetEncoder("resnet18", out_channels=16, num_levels=2)
    feats = enc(torch.rand(1, 3, 64, 128))
    assert [f.shape[-2:] for f in feats] == [(8, 16), (4, 8)]


def test_decoder_and_head_keep_resolution() -> None:
    dec = BEVDecoder(16, 8)
    x = torch.randn(2, 16, 15, 17)  # odd sizes must survive the U-Net
    out = SegmentationHead(8, 3)(dec(x))
    assert out.shape == (2, 3, 15, 17)


def test_iou_metric() -> None:
    m = IoUMetric(3)
    pred = torch.tensor([0, 1, 1, 2, 0])
    tgt = torch.tensor([0, 1, 2, 2, 1])
    m.update(pred, tgt)
    iou = m.per_class()
    assert torch.allclose(iou, torch.tensor([1 / 2, 1 / 3, 1 / 2]))
    assert abs(m.mean_foreground() - (1 / 3 + 1 / 2) / 2) < 1e-6
    m.reset()
    assert m.per_class().isnan().all() and m.mean_foreground() == 0.0
