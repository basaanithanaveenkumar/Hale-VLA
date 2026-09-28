"""Semantic object categories of the synthetic world."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectClass:
    """A box-shaped object category.

    Attributes:
        name: human readable name.
        label: integer class id (``0`` is reserved for background / empty).
        color: base RGB colour in ``[0, 1]``.
        length, width, height: ``(min, max)`` size ranges in metres.
    """

    name: str
    label: int
    color: tuple[float, float, float]
    length: tuple[float, float]
    width: tuple[float, float]
    height: tuple[float, float]


VEHICLE = ObjectClass("vehicle", 1, (0.85, 0.15, 0.12), (3.6, 4.6), (1.7, 2.0), (1.4, 1.8))
PEDESTRIAN = ObjectClass("pedestrian", 2, (0.12, 0.35, 0.9), (0.8, 1.1), (0.8, 1.1), (1.6, 1.9))
DEFAULT_CLASSES: tuple[ObjectClass, ...] = (VEHICLE, PEDESTRIAN)
