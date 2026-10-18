"""Set ``HaloVLMConfig`` fields by name: YAML architecture keys and ``--model_option KEY=VALUE``."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from config.model_config import HaloVLMConfig
from exceptions import HaloVLAConfigError

_FIELDS = {f.name: str(f.type) for f in dataclasses.fields(HaloVLMConfig)}


def _parse(kind: str, raw: str) -> Any:
    """``raw`` typed by the dataclass annotation ``kind`` (``"int | None"``, ``"tuple[int, ...]"`` ...)."""
    if raw.lower() in ("none", "null"):
        return None
    if kind.startswith(("tuple", "list")):
        if raw.lstrip().startswith(("[", "(")):  # JSON-style, also nested: [[-16, 16], [-16, 16]]
            return json.loads(raw.replace("(", "[").replace(")", "]"))
        items = [x.strip() for x in raw.strip("[]()").split(",") if x.strip()]
        item_kind = "int" if "int" in kind else "str"
        return [_parse(item_kind, x) for x in items]
    if kind.startswith("dict"):
        return json.loads(raw)
    if kind.startswith("bool"):
        if raw.lower() not in ("1", "0", "true", "false", "yes", "no"):
            raise HaloVLAConfigError(f"expected a boolean, got {raw!r}")
        return raw.lower() in ("1", "true", "yes")
    if kind.startswith("int"):
        return int(raw)
    if kind.startswith("float"):
        return float(raw)
    return raw


def parse_model_options(items: list[str] | None) -> dict[str, Any]:
    """``["emb_dim=256", "use_moe=false"]`` -> typed ``{field: value}`` for :class:`HaloVLMConfig`."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in _FIELDS:
            raise HaloVLAConfigError(f"bad model option {item!r}; use KEY=VALUE with KEY a HaloVLMConfig field")
        out[key] = _parse(_FIELDS[key], raw)
    return out


def _as_tuple(value: Any) -> Any:
    """Lists (also nested, e.g. ``bev_bounds``) -> tuples, as the dataclass declares them."""
    return tuple(_as_tuple(v) for v in value) if isinstance(value, (list, tuple)) else value


def apply_model_overrides(config: HaloVLMConfig, overrides: dict[str, Any]) -> HaloVLMConfig:
    """Set ``overrides`` on ``config`` (lists become tuples for tuple fields) and re-validate it."""
    unknown = set(overrides) - set(_FIELDS)
    if unknown:
        raise HaloVLAConfigError(f"unknown HaloVLMConfig fields: {sorted(unknown)}")
    for key, value in overrides.items():
        if _FIELDS[key].startswith("tuple") and value is not None:
            value = _as_tuple(value)
        setattr(config, key, value)
    config.__post_init__()
    return config
