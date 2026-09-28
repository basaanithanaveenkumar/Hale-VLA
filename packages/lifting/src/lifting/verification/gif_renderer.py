"""Animated GIF of a verification run: cameras, ground truth and every lifter's BEV over training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from lifting.data import SceneBatch
from lifting.verification.lifter_report import LifterReport
from lifting.verification.verification_report import VerificationReport

PALETTE = np.array([[38, 40, 46], [230, 60, 45], [50, 120, 245]], dtype=np.uint8)
BACKGROUND = (18, 19, 23)
TEXT = (235, 235, 235)
PASS_COLOR = (90, 210, 120)
FAIL_COLOR = (240, 90, 80)


class VerificationGifRenderer:
    """Renders one frame per training snapshot plus two summary frames.

    Args:
        tile: pixel size of a BEV tile.
        columns: tiles per row.
    """

    def __init__(self, tile: int = 128, columns: int = 4) -> None:
        self.tile = tile
        self.columns = columns
        self.label_height = 30
        try:
            self.font = ImageFont.load_default(size=12)
        except TypeError:  # Pillow < 10.1
            self.font = ImageFont.load_default()

    # ------------------------------------------------------------------ tiles
    @staticmethod
    def to_bev(labels: Tensor) -> np.ndarray:
        """``(X, Y)`` or ``(X, Y, Z)`` labels -> ``(X, Y)``; voxels are projected by max over height."""
        arr = labels.cpu().numpy()
        return arr.max(-1) if arr.ndim == 3 else arr

    def bev_tile(self, labels: np.ndarray) -> Image.Image:
        """Colour a BEV label map with x (forward) pointing up and y (left) pointing left."""
        rgb = PALETTE[np.clip(labels, 0, len(PALETTE) - 1)][::-1, ::-1]
        img = Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (self.tile, self.tile), Image.NEAREST
        )
        draw = ImageDraw.Draw(img)
        c = self.tile // 2
        draw.rectangle([c - 3, c - 5, c + 3, c + 5], fill=(255, 255, 255))  # ego vehicle
        return img

    def labelled(
        self, tile: Image.Image, title: str, subtitle: str = "", color: tuple = TEXT
    ) -> Image.Image:
        canvas = Image.new("RGB", (self.tile, self.tile + self.label_height), BACKGROUND)
        canvas.paste(tile, (0, self.label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), title, fill=TEXT, font=self.font)
        draw.text((4, 15), subtitle, fill=color, font=self.font)
        return canvas

    def camera_strip(self, images: Tensor, width: int) -> Image.Image:
        """``(N, 3, H, W)`` images side by side, scaled to ``width``."""
        arr = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        strip = Image.fromarray(np.concatenate(list(arr), axis=1))
        height = round(strip.height * width / strip.width)
        return strip.resize((width, height), Image.BILINEAR)

    # ------------------------------------------------------------------ frames
    def _compose(self, title: str, cameras: Image.Image, tiles: list[Image.Image]) -> Image.Image:
        rows = -(-len(tiles) // self.columns)
        tile_h = self.tile + self.label_height
        width = self.columns * self.tile
        height = 26 + cameras.height + 4 + rows * tile_h
        frame = Image.new("RGB", (width, height), BACKGROUND)
        ImageDraw.Draw(frame).text((6, 6), title, fill=TEXT, font=self.font)
        frame.paste(cameras, (0, 26))
        for i, t in enumerate(tiles):
            r, c = divmod(i, self.columns)
            frame.paste(t, (c * self.tile, 26 + cameras.height + 4 + r * tile_h))
        return frame

    @staticmethod
    def _status(report: LifterReport) -> tuple[str, tuple]:
        if not report.checks:
            return f"mIoU {report.miou:.2f} (baseline)", TEXT
        ok = report.passed
        return (
            f"mIoU {report.miou:.2f} {'PASS' if ok else 'FAIL'}",
            PASS_COLOR if ok else FAIL_COLOR,
        )

    def frames(self, report: VerificationReport, batch: SceneBatch) -> list[Image.Image]:
        rows = [report.baseline, *report.lifters]
        cameras = self.camera_strip(batch.images[0], self.columns * self.tile)
        gt_bev = self.labelled(
            self.bev_tile(self.to_bev(batch.bev_labels[0])), "ground truth", "BEV (x up, y left)"
        )
        gt_occ = self.labelled(
            self.bev_tile(self.to_bev(batch.occupancy[0])), "ground truth 3D", "occupancy top view"
        )
        has_occ = any(r.task == "occupancy" for r in rows)

        def gt_tiles() -> list[Image.Image]:
            return [gt_bev, gt_occ] if has_occ else [gt_bev]

        frames = []
        num_snapshots = min(len(r.snapshots) for r in rows)
        for k in range(num_snapshots):
            step = rows[0].snapshots[k][0]
            tiles = gt_tiles() + [
                self.labelled(
                    self.bev_tile(self.to_bev(r.snapshots[k][1])),
                    r.name,
                    f"step {r.snapshots[k][0]}",
                )
                for r in rows
            ]
            frames.append(self._compose(f"lifting verifier | training step {step}", cameras, tiles))

        final = gt_tiles() + [
            self.labelled(self.bev_tile(self.to_bev(r.snapshots[-1][1])), r.name, *self._status(r))
            for r in rows
        ]
        verdict = "PASS" if report.passed else "FAIL"
        frames.append(
            self._compose(f"lifting verifier | final | overall {verdict}", cameras, final)
        )

        corrupted = gt_tiles() + [
            self.labelled(
                self.bev_tile(self.to_bev(r.corrupted_prediction)),
                r.name,
                f"mIoU {r.miou_corrupted:.2f} bad calib",
                FAIL_COLOR if r.uses_geometry else TEXT,
            )
            for r in rows
            if r.corrupted_prediction is not None
        ]
        frames.append(
            self._compose("extrinsics rotated -> geometric lifters must break", cameras, corrupted)
        )
        return frames

    def save(
        self, report: VerificationReport, batch: SceneBatch, path: str | Path, frame_ms: int = 350
    ) -> Path:
        frames = self.frames(report, batch)
        durations = [frame_ms] * (len(frames) - 2) + [2500, 2500]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
        )
        return path
