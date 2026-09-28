import pytest
import torch

from lifting.ops import voxel_pooling


def naive(features, index, valid, cells, reduce):
    b, _, c = features.shape
    out = torch.zeros(b, cells, c)
    count = torch.zeros(b, cells)
    for bi in range(b):
        for pi in range(features.shape[1]):
            if valid[bi, pi]:
                out[bi, index[bi, pi]] += features[bi, pi]
                count[bi, index[bi, pi]] += 1
    if reduce == "mean":
        out = out / count.clamp(min=1)[..., None]
    return out


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_matches_naive(reduce: str) -> None:
    feats = torch.randn(3, 200, 5)
    index = torch.randint(0, 17, (3, 200))
    valid = torch.rand(3, 200) > 0.3
    out = voxel_pooling(feats, index, valid, 17, reduce)
    assert torch.allclose(out, naive(feats, index, valid, 17, reduce), atol=1e-5)


def test_batches_do_not_mix() -> None:
    feats = torch.ones(2, 4, 1)
    index = torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2]])
    out = voxel_pooling(feats, index, torch.ones(2, 4, dtype=torch.bool), 3)
    assert out[0, :, 0].tolist() == [2, 2, 0]
    assert out[1, :, 0].tolist() == [0, 0, 4]


def test_gradient_flows_only_to_valid_points() -> None:
    feats = torch.randn(1, 6, 2, requires_grad=True)
    valid = torch.tensor([[True, False, True, False, True, True]])
    voxel_pooling(feats, torch.zeros(1, 6, dtype=torch.long), valid, 2).sum().backward()
    assert torch.equal(feats.grad[0, :, 0], valid[0].float())


def test_bad_reduce() -> None:
    with pytest.raises(ValueError):
        voxel_pooling(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            1,
            "max",
        )
