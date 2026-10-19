"""Notebook helpers for ``notebooks/open_vlas``: :mod:`vla_kit.plots` and :mod:`vla_kit.diagrams`; :func:`setup_paths` makes the
``open_vlas`` and ``pi_family`` packages (in ``src/Halo_VLA``) importable from a notebook."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> Path:
    """Find the repo root from the current directory and put ``src/Halo_VLA`` and the notebook folder on ``sys.path``."""
    here = Path.cwd().resolve()
    for root in (here, *here.parents):
        if (root / "src" / "Halo_VLA" / "open_vlas").is_dir():
            for p in (root / "src" / "Halo_VLA", root / "notebooks" / "open_vlas"):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            return root
    raise RuntimeError("run the notebook from inside the Hale-VLA repository")
