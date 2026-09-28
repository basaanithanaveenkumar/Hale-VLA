"""Procedural 3D synthetic scenes with exact camera geometry and labels."""

from lifting.data.camera_rig import CameraRig
from lifting.data.label_rasterizer import LabelRasterizer
from lifting.data.object_class import DEFAULT_CLASSES, PEDESTRIAN, VEHICLE, ObjectClass
from lifting.data.object_sampler import ObjectSampler
from lifting.data.ray_cast_renderer import RayCastRenderer, RenderResult
from lifting.data.scene_batch import SceneBatch
from lifting.data.scene_objects import SceneObjects
from lifting.data.synthetic_scene_dataset import SyntheticSceneDataset

__all__ = [
    "DEFAULT_CLASSES",
    "PEDESTRIAN",
    "VEHICLE",
    "CameraRig",
    "LabelRasterizer",
    "ObjectClass",
    "ObjectSampler",
    "RayCastRenderer",
    "RenderResult",
    "SceneBatch",
    "SceneObjects",
    "SyntheticSceneDataset",
]
