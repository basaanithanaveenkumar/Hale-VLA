"""Notebook helpers for ``notebooks/vla_foundations``: :mod:`fnd_kit.diagrams` and :mod:`fnd_kit.plots` (and, through
:func:`setup_paths`, the shared ``vla_kit`` of ``notebooks/open_vlas``)."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> Path:
    """Find the repo root and put ``src/Halo_VLA`` and both notebook kits on ``sys.path``."""
    here = Path.cwd().resolve()
    for root in (here, *here.parents):
        if (root / "src" / "Halo_VLA" / "vla_foundations").is_dir():
            for p in (
                root / "src" / "Halo_VLA",
                root / "notebooks" / "open_vlas",
                root / "notebooks" / "vla_foundations",
            ):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            return root
    raise RuntimeError("run the notebook from inside the Hale-VLA repository")
