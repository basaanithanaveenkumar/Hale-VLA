import pytest
import torch

from lifting.attention import (
    CrossViewHybridAttention,
    MSDeformableAttention2D,
    MSDeformableAttention3D,
    PillarDeformableAttention,
    SpatialCrossAttention,
)


def test_ms_deformable_attention_2d_shapes_and_grads() -> None:
    shapes = [(8, 12), (4, 6)]
    attn = MSDeformableAttention2D(32, 4, num_levels=2, num_points=3)
    q = torch.randn(2, 10, 32)
    v = torch.randn(2, sum(h * w for h, w in shapes), 32, requires_grad=True)
    out = attn(q, v, torch.rand(2, 10, 2), shapes)
    assert out.shape == (2, 10, 32)
    out.sum().backward()
    assert v.grad.abs().sum() > 0
    assert attn.sampling_offsets.weight.grad is not None
    # per-level reference points are accepted too
    assert attn(q, v, torch.rand(2, 10, 2, 2), shapes).shape == (2, 10, 32)


def test_offset_initialisation_spreads_points() -> None:
    attn = MSDeformableAttention2D(32, 4, num_levels=1, num_points=2)
    bias = attn.sampling_offsets.bias.view(4, 1, 2, 2)
    # point i of each head is i + 1 cells from the reference, heads look in different directions
    assert torch.allclose(bias[:, 0, 1], 2 * bias[:, 0, 0])
    assert len({tuple(d.sign().tolist()) for d in bias[:, 0, 0]}) == 4


def test_ms_deformable_attention_3d_shapes_and_grads() -> None:
    shapes = [(4, 8, 8), (2, 4, 4)]
    attn = MSDeformableAttention3D(32, 4, num_levels=2, num_points=3)
    q = torch.randn(2, 7, 32)
    v = torch.randn(2, sum(d * h * w for d, h, w in shapes), 32, requires_grad=True)
    out = attn(q, v, torch.rand(2, 7, 3), shapes)
    assert out.shape == (2, 7, 32)
    out.square().sum().backward()
    assert v.grad.abs().sum() > 0
    # offsets / weights start at zero weight but receive gradient
    assert attn.sampling_offsets.weight.grad.abs().sum() > 0
    assert attn.attention_weights.weight.grad.abs().sum() > 0


def test_pillar_attention_requires_divisible_points() -> None:
    with pytest.raises(ValueError):
        PillarDeformableAttention(32, 4, 1, num_points=6, num_anchors=4)
    attn = PillarDeformableAttention(32, 4, 1, num_points=8, num_anchors=4)
    out = attn(torch.randn(2, 5, 32), torch.randn(2, 6 * 8, 32), torch.rand(2, 5, 4, 2), [(6, 8)])
    assert out.shape == (2, 5, 32)


def test_spatial_cross_attention_ignores_cameras_that_cannot_see_the_query() -> None:
    sca = SpatialCrossAttention(32, 4, 1, num_points=4, num_anchors=2)
    q = torch.randn(1, 6, 32)
    values = torch.randn(1, 3, 20, 32)
    ref = torch.rand(1, 3, 6, 2, 2)
    visible = torch.zeros(1, 3, 6, 2, dtype=torch.bool)
    visible[:, 0, :3] = True  # queries 0-2 seen by camera 0 only
    out1 = sca(q, values, [(4, 5)], ref, visible)
    values2 = values.clone()
    values2[:, 1:] = torch.randn(1, 2, 20, 32)  # change the cameras that see nothing
    out2 = sca(q, values2, [(4, 5)], ref, visible)
    assert torch.allclose(out1, out2, atol=1e-6)
    # queries seen by no camera get only the output bias
    assert torch.allclose(out1[0, 3:], sca.output_proj.bias.expand(3, -1), atol=1e-6)
    values3 = values.clone()
    values3[:, 0] = torch.randn(1, 20, 32)
    assert not torch.allclose(sca(q, values3, [(4, 5)], ref, visible)[0, :3], out1[0, :3])


def test_cross_view_hybrid_attention() -> None:
    shapes = [(4, 4), (4, 2), (4, 2)]
    tokens = sum(h * w for h, w in shapes)
    attn = CrossViewHybridAttention(32, 4, num_points=2)
    out = attn(
        torch.randn(2, tokens, 32), torch.randn(2, tokens, 32), shapes, torch.rand(2, tokens, 3, 2)
    )
    assert out.shape == (2, tokens, 32)
