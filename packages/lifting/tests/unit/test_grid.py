import torch

from lifting.geometry import DepthBins, GridSpec


def test_voxel_centers_and_index_agree(grid: GridSpec) -> None:
    centers = grid.voxel_centers()
    assert centers.shape == (*grid.resolution, 3)
    idx = grid.to_index(centers)
    expected = torch.stack(
        torch.meshgrid(*(torch.arange(n) + 0.5 for n in grid.resolution), indexing="ij"), -1
    )
    assert torch.allclose(idx, expected, atol=1e-5)
    assert grid.contains(centers).all()


def test_normalized_coordinates_span_minus_one_to_one(grid: GridSpec) -> None:
    corners = torch.tensor(
        [
            [grid.x_bounds[0], grid.y_bounds[0], grid.z_bounds[0]],
            [grid.x_bounds[1], grid.y_bounds[1], grid.z_bounds[1]],
        ]
    )
    assert torch.allclose(grid.to_normalized(corners), torch.tensor([[-1.0] * 3, [1.0] * 3]))


def test_flat_indices(grid: GridSpec) -> None:
    centers = grid.voxel_centers().view(-1, 3)
    flat, valid = grid.flat_voxel_index(centers)
    assert valid.all()
    assert torch.equal(flat, torch.arange(centers.shape[0]))
    bev, _ = grid.flat_bev_index(grid.voxel_centers()[:, :, 0].reshape(-1, 3))
    assert torch.equal(bev, torch.arange(grid.num_cells_bev))
    outside = torch.tensor([[100.0, 0.0, 0.0]])
    assert not grid.flat_voxel_index(outside)[1].any()


def test_line_points_lie_on_lines(grid: GridSpec) -> None:
    xs, ys, zs = (grid.axis_centers(a) for a in range(3))
    pillars = grid.line_points(axis=2, num_anchors=5)
    assert pillars.shape == (16, 16, 5, 3)
    assert torch.allclose(pillars[3, 7, :, 0], xs[3].expand(5))
    assert torch.allclose(pillars[3, 7, :, 1], ys[7].expand(5))
    assert (pillars[..., 2] > grid.z_bounds[0]).all() and (pillars[..., 2] < grid.z_bounds[1]).all()

    xz = grid.line_points(axis=1, num_anchors=6)
    assert xz.shape == (16, 4, 6, 3)
    assert torch.allclose(xz[2, 1, :, 0], xs[2].expand(6))
    assert torch.allclose(xz[2, 1, :, 2], zs[1].expand(6))

    yz = grid.line_points(axis=0, num_anchors=6)
    assert yz.shape == (16, 4, 6, 3)
    assert torch.allclose(yz[5, 2, :, 1], ys[5].expand(6))
    assert torch.allclose(yz[5, 2, :, 2], zs[2].expand(6))


def test_depth_bins() -> None:
    bins = DepthBins(1.0, 9.0, 8)
    centers = bins.centers()
    assert torch.allclose(centers, torch.arange(8) + 1.5)
    # bin centres map onto grid_sample(align_corners=False) pixel centres
    assert torch.allclose(bins.to_normalized(centers), (2 * (torch.arange(8) + 0.5) / 8) - 1)
    assert bins.contains(torch.tensor([1.0, 8.99])).all()
    assert not bins.contains(torch.tensor([0.5, 9.0])).any()
