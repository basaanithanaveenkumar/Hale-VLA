"""2D -> BEV / 3D lifters. Importing this package registers all of them in :data:`LIFTERS`.

| name          | class                    | origin (simple_bev/nets, TPVFormer) |
|---------------|--------------------------|-------------------------------------|
| ``simple_bev``| BilinearSamplingLifter   | segnet.py                           |
| ``depth_warp``| DepthWarpLifter          | liftnet.py                          |
| ``lift_splat``| LiftSplatLifter          | liftnet2.py                         |
| ``bevformer`` | BEVFormerLifter          | bevformernet.py / bevformernet2.py  |
| ``tiim``      | PolarRayLifter           | tiimnet.py                          |
| ``tpvformer`` | TPVFormerLifter          | wzzheng/TPVFormer                   |
| ``mlp_view``  | MLPViewLifter            | geometry-free baseline (VPN)        |
"""

from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.lifters.bevformer import BEVFormerLifter
from lifting.lifters.bilinear_sampling_lifter import BilinearSamplingLifter
from lifting.lifters.depth_head import DepthHead
from lifting.lifters.depth_warp_lifter import DepthWarpLifter
from lifting.lifters.lift_splat_lifter import LiftSplatLifter
from lifting.lifters.mlp_view_lifter import MLPViewLifter
from lifting.lifters.polar_ray_lifter import PolarRayLifter
from lifting.lifters.tpvformer import TPVAggregator, TPVFormerLifter, TPVPlanes, TPVReferencePoints

__all__ = [
    "LIFTERS",
    "BEVFormerLifter",
    "BaseLifter",
    "BilinearSamplingLifter",
    "DepthHead",
    "DepthWarpLifter",
    "LiftSplatLifter",
    "MLPViewLifter",
    "PolarRayLifter",
    "TPVAggregator",
    "TPVFormerLifter",
    "TPVPlanes",
    "TPVReferencePoints",
]
