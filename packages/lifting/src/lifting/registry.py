"""A tiny name -> class registry so new components plug in without editing core code."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string names to classes (or factories).

    Example::

        LIFTERS = Registry("lifter")

        @LIFTERS.register("my_lifter")
        class MyLifter(BaseLifter): ...

        lifter = LIFTERS.build("my_lifter", grid=grid, in_channels=64, out_channels=64)
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._entries[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}") from None

    def build(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
