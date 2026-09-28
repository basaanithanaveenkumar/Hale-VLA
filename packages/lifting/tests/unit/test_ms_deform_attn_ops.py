"""The pure-PyTorch deformable attention ops against explicit loop implementations."""

import math

import pytest
import torch
import torch.nn.functional as F

from lifting.ops import multi_scale_deformable_attn_2d, multi_scale_deformable_attn_3d


def bilinear(img: torch.Tensor, x: float, y: float) -> torch.Tensor:
    """Zero-padded bilinear read of ``img (C, H, W)`` at normalised ``(x, y)`` in [0, 1]."""
    c, h, w = img.shape
    px, py = x * w - 0.5, y * h - 0.5
    x0, y0 = math.floor(px), math.floor(py)
    out = torch.zeros(c, dtype=img.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            wgt = (1 - abs(px - xi)) * (1 - abs(py - yi))
            if 0 <= xi < w and 0 <= yi < h:
                out += wgt * img[:, yi, xi]
    return out


def trilinear(vol: torch.Tensor, x: float, y: float, z: float) -> torch.Tensor:
    c, d, h, w = vol.shape
    px, py, pz = x * w - 0.5, y * h - 0.5, z * d - 0.5
    x0, y0, z0 = math.floor(px), math.floor(py), math.floor(pz)
    out = torch.zeros(c, dtype=vol.dtype)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi, zi = x0 + dx, y0 + dy, z0 + dz
                wgt = (1 - abs(px - xi)) * (1 - abs(py - yi)) * (1 - abs(pz - zi))
                if 0 <= xi < w and 0 <= yi < h and 0 <= zi < d:
                    out += wgt * vol[:, zi, yi, xi]
    return out


SHAPES_2D = [(5, 7), (3, 4)]
SHAPES_3D = [(3, 4, 5), (2, 2, 3)]


def naive_2d(value, shapes, loc, attn):
    b, _, m, dim = value.shape
    q, lv, p = loc.shape[1], loc.shape[3], loc.shape[4]
    levels = value.split([h * w for h, w in shapes], 1)
    out = torch.zeros(b, q, m, dim, dtype=value.dtype)
    for bi in range(b):
        for qi in range(q):
            for mi in range(m):
                for li in range(lv):
                    h, w = shapes[li]
                    img = levels[li][bi, :, mi].T.reshape(dim, h, w)
                    for pi in range(p):
                        x, y = loc[bi, qi, mi, li, pi].tolist()
                        out[bi, qi, mi] += attn[bi, qi, mi, li, pi] * bilinear(img, x, y)
    return out.flatten(2)


def naive_3d(value, shapes, loc, attn):
    b, _, m, dim = value.shape
    q, lv, p = loc.shape[1], loc.shape[3], loc.shape[4]
    levels = value.split([d * h * w for d, h, w in shapes], 1)
    out = torch.zeros(b, q, m, dim, dtype=value.dtype)
    for bi in range(b):
        for qi in range(q):
            for mi in range(m):
                for li in range(lv):
                    d, h, w = shapes[li]
                    vol = levels[li][bi, :, mi].T.reshape(dim, d, h, w)
                    for pi in range(p):
                        x, y, z = loc[bi, qi, mi, li, pi].tolist()
                        out[bi, qi, mi] += attn[bi, qi, mi, li, pi] * trilinear(vol, x, y, z)
    return out.flatten(2)


def random_inputs(shapes, coord_dims, dtype=torch.float64):
    b, q, m, dim, p = 2, 3, 2, 4, 3
    s = sum(math.prod(sh) for sh in shapes)
    value = torch.randn(b, s, m, dim, dtype=dtype)
    loc = (
        torch.rand(b, q, m, len(shapes), p, coord_dims, dtype=dtype) * 1.2 - 0.1
    )  # includes borders
    attn = torch.rand(b, q, m, len(shapes), p, dtype=dtype)
    attn = attn / attn.sum((-1, -2), keepdim=True)
    return value, loc, attn


def test_2d_matches_naive_loop() -> None:
    value, loc, attn = random_inputs(SHAPES_2D, 2)
    out = multi_scale_deformable_attn_2d(value, SHAPES_2D, loc, attn)
    assert out.shape == (2, 3, 8)
    assert torch.allclose(out, naive_2d(value, SHAPES_2D, loc, attn), atol=1e-10)


def test_3d_matches_naive_loop() -> None:
    value, loc, attn = random_inputs(SHAPES_3D, 3)
    out = multi_scale_deformable_attn_3d(value, SHAPES_3D, loc, attn)
    assert out.shape == (2, 3, 8)
    assert torch.allclose(out, naive_3d(value, SHAPES_3D, loc, attn), atol=1e-10)


def test_2d_single_point_equals_grid_sample() -> None:
    h, w = 6, 9
    img = torch.randn(1, 4, h, w)
    value = img.flatten(2).transpose(1, 2)[:, :, None]  # (1, HW, 1, 4)
    loc = torch.rand(1, 10, 1, 1, 1, 2)
    out = multi_scale_deformable_attn_2d(value, [(h, w)], loc, torch.ones(1, 10, 1, 1, 1))
    ref = F.grid_sample(img, 2 * loc.view(1, 1, 10, 2) - 1, align_corners=False)[0, :, 0].T
    assert torch.allclose(out[0], ref, atol=1e-6)


@pytest.mark.parametrize(
    "op,shapes,dims",
    [
        (multi_scale_deformable_attn_2d, SHAPES_2D, 2),
        (multi_scale_deformable_attn_3d, SHAPES_3D, 3),
    ],
)
def test_constant_field_is_preserved(op, shapes, dims) -> None:
    value, _, attn = random_inputs(shapes, dims)
    value = torch.full_like(value, 2.5)
    loc = 0.25 + 0.5 * torch.rand(2, 3, 2, len(shapes), 3, dims, dtype=torch.float64)
    # interior points of a constant field: weights sum to 1 -> output is the constant
    # (coarse levels have few cells, so stay away from the zero-padded border)
    loc = loc.clamp(0.35, 0.65)
    out = op(value, shapes, loc, attn)
    assert torch.allclose(out, torch.full_like(out, 2.5))


@pytest.mark.parametrize(
    "op,shapes,dims",
    [
        (multi_scale_deformable_attn_2d, SHAPES_2D, 2),
        (multi_scale_deformable_attn_3d, SHAPES_3D, 3),
    ],
)
def test_gradcheck(op, shapes, dims) -> None:
    value, loc, attn = random_inputs(shapes, dims)
    loc = loc.clamp(0.05, 0.95)  # avoid kinks exactly at the border
    value.requires_grad_()
    loc.requires_grad_()
    attn.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda v, loc_, a: op(v, shapes, loc_, a), (value, loc, attn), atol=1e-6
    )


def test_level_count_mismatch_raises() -> None:
    value, loc, attn = random_inputs(SHAPES_2D, 2)
    with pytest.raises(ValueError):
        multi_scale_deformable_attn_2d(value, SHAPES_2D[:1], loc, attn)
