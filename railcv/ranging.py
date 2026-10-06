"""Monocular range from the ego path, using the rail gauge as a built-in ruler.

At any image row, the two ego rails sit at (roughly) the same depth Z, so their pixel
separation is   dx = fx * G / Z   ->   Z = fx * G / dx
with G = rail center-to-center spacing (RailSem19 / TEP-Net rails are centerlines).

On level, straight track dx is also linear in the image row y:
    dx = (fx * G) / (fy * H) * (y - y_horizon)
so a robust line fit of dx vs y (a) averages out per-anchor pixel noise and
(b) gives two self-checks for free: the horizon row and the camera height H.
If H disagrees with the tape-measured mount height, the intrinsics or the gauge are off.

Range error grows with Z^2: one pixel of dx error -> dZ ~= Z^2 / (fx * G).
"""

from dataclasses import dataclass, field

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # ranging math still works without undistortion
    cv2 = None

MPH = 0.44704  # m/s per mph


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    gauge_m: float = 1.505
    mount_height_m: float | None = None

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            c = yaml.safe_load(f)
        intr = c["intrinsics"]
        return cls(
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            dist=np.array(intr.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64),
            gauge_m=float(c["track"]["gauge_center_to_center_m"]),
            mount_height_m=c.get("mount", {}).get("height_m"),
        )

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        if cv2 is None or not np.any(self.dist) or len(pts) == 0:
            return pts
        und = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist, P=self.K)
        return und.reshape(-1, 2)


@dataclass
class RangeResult:
    y: np.ndarray  # image row of each anchor (undistorted px)
    dx: np.ndarray  # rail separation per anchor (px)
    z_raw: np.ndarray  # per-anchor range from dx alone (m)
    valid: np.ndarray  # anchors usable for ranging
    fit: tuple | None  # (a, b) of dx = a*y + b, or None
    clear_m: float | None  # distance to the far end of the detected ego path
    clear_err_m: float | None  # +- range error for 1 px of dx error at that distance
    horizon_y: float | None
    cam_height_m: float | None

    fxg: float = 0.0  # fx * gauge, the ranging constant

    def z_at_row(self, y):
        """Fitted range (m) at image row(s) y; inf above the horizon."""
        if self.fit is None:
            return None
        a, b = self.fit
        d = a * np.asarray(y, dtype=np.float64) + b
        return np.divide(self.fxg, d, out=np.full_like(d, np.inf), where=d > 0)

    def row_at_z(self, z):
        """Image row where the fitted path reaches range z (for drawing distance ticks)."""
        if self.fit is None:
            return None
        a, b = self.fit
        return (self.fxg / z - b) / a


def estimate_range(left, right, cam: CameraModel, crop=None, min_dx_px=4.0, edge_px=2.0) -> RangeResult:
    """left/right: (N, 2) absolute pixel rail points from the same anchor rows."""
    n = min(len(left), len(right))
    fxg = cam.fx * cam.gauge_m
    if n < 2:
        return RangeResult(np.empty(0), np.empty(0), np.empty(0), np.empty(0, bool), None, None, None, None, None, fxg)
    left, right = np.asarray(left[:n], float), np.asarray(right[:n], float)

    # TEP-Net clips rail x to the crop; clipped anchors give a wrong separation
    valid = np.ones(n, bool)
    if crop is not None:
        x0, _, x1, _ = crop
        for r in (left, right):
            valid &= (r[:, 0] > x0 + edge_px) & (r[:, 0] < x1 - edge_px)

    lu, ru = cam.undistort(left), cam.undistort(right)
    y = 0.5 * (lu[:, 1] + ru[:, 1])
    dx = ru[:, 0] - lu[:, 0]
    valid &= dx > min_dx_px

    z_raw = np.divide(fxg, dx, out=np.full(n, np.inf), where=dx > 0)

    fit = None
    if valid.sum() >= 6:
        yy, dd = y[valid], dx[valid]
        a, b = np.polyfit(yy, dd, 1)
        res = dd - (a * yy + b)
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        keep = np.abs(res) < 3.0 * 1.4826 * mad + 0.5
        if keep.sum() >= 6:
            a, b = np.polyfit(yy[keep], dd[keep], 1)
        if a > 0:  # rails must widen toward the bottom of the image
            fit = (float(a), float(b))

    clear = err = horizon = height = None
    if fit is not None:
        a, b = fit
        y_far = y[valid].min()
        d_far = a * y_far + b
        if d_far > 0:
            clear = fxg / d_far
            err = clear**2 / fxg
        horizon = -b / a
        height = cam.fx * cam.gauge_m / (cam.fy * a)
    elif valid.any():
        i = np.argmin(np.where(valid, y, np.inf))
        clear = float(z_raw[i])
        err = clear**2 / fxg

    return RangeResult(y, dx, z_raw, valid, fit, clear, err, horizon, height, fxg)


def stopping_distance(speed_mps, decel_mps2, latency_s):
    """Reaction travel + braking distance: v*t + v^2 / (2a)."""
    return speed_mps * latency_s + speed_mps**2 / (2.0 * decel_mps2)
