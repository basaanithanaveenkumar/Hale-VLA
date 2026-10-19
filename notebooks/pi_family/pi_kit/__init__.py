"""Notebook helpers for ``notebooks/pi_family``: :mod:`pi_kit.plots` (data plots) and :mod:`pi_kit.diagrams`
(schematic figures). :func:`setup_paths` makes ``pi_family`` (in ``src/Halo_VLA``) importable from a notebook."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> Path:
    """Find the repo root from the current directory and put ``src/Halo_VLA`` on ``sys.path``."""
    here = Path.cwd().resolve()
    for root in (here, *here.parents):
        if (root / "src" / "Halo_VLA" / "pi_family").is_dir():
            for p in (root / "src" / "Halo_VLA", root / "notebooks" / "pi_family"):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            return root
    raise RuntimeError("run the notebook from inside the Hale-VLA repository")
