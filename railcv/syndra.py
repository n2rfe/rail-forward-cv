"""SynDRA / SynDRA-Gen access and ground-truth extraction for ego-path / ranging evaluation.

Two layouts are supported:

  SynDRA dataset (website download)
    <root>/Scenario_0/HV/Sunny/Morning/RGBCamera_0/Bin_folder/Sce0_sun_mor_000001.bin
                                      /SSCamera_0/...
    Convert with SynDRA's Utils/*_bin_to_png.py first; PNGs under a camera folder are used.

  SynDRA-Gen (the Unreal Engine generator app, github.com/DAmicoGlc/SynDRA-Gen)
    <app>/DataAcquired/<timestamp>/RGBCamera/Bin_folder/frame_000000.bin     W*H*3 bytes, BGR
                                  /SSCamera/Bin_folder/frame_000000.bin      W*H bytes, one label id per pixel
                                  /DepthCamera/Bin_folder/frame_000000.bin   W*H*2 bytes, 16-bit (high byte first)
    (stereo rigs use RGBCamera_Left / _Right etc.). The .bin files are read directly; the image
    size is inferred from the file size.
"""

import math
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from .ranging import CameraModel

_IDX = re.compile(r"(\d+)$")

# camera folder names per camera index: dataset layout first, then SynDRA-Gen mono / stereo
_CAM_NAMES = {
    0: ("{k}Camera_0", "{k}Camera", "{k}Camera_Left"),
    1: ("{k}Camera_1", "{k}Camera_Right"),
}
# resolutions tried when a .bin size doesn't match the configured one
_PRESETS = [(1920, 1080), (1280, 960), (1280, 720), (2560, 1440), (3840, 2160), (1600, 900), (1024, 768),
            (640, 480), (960, 540), (800, 600), (1920, 1200), (2048, 1536), (1280, 1024), (1024, 1024), (512, 512)]


# --------------------------------------------------------------------------- config / camera
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def camera_model(cfg, image_shape=None) -> CameraModel:
    """Pinhole model from the horizontal FOV (square pixels, no distortion, centered principal point).

    image_shape (H, W) overrides the configured resolution, so the model always matches the frames.
    """
    c = cfg["camera"]
    h, w = image_shape[:2] if image_shape is not None else (c["height"], c["width"])
    fx = (w / 2.0) / math.tan(math.radians(c["hfov_deg"]) / 2.0)
    hm = c.get("height_m")
    return CameraModel(
        fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0,
        gauge_m=float(cfg["track"]["gauge_center_to_center_m"]),
        mount_height_m=float(hm) if hm is not None else None,
    )


# --------------------------------------------------------------------------- discovery
@dataclass
class Sequence:
    path: str
    scenario: str
    visibility: str  # HV / LV (dataset layout only)
    weather: str
    time: str
    rgb_dir: str = ""
    ss_dir: str = ""
    depth_dir: str = ""  # "" when the sequence has no depth camera
    generated: bool = False  # True for SynDRA-Gen output

    @property
    def name(self):
        if self.generated:
            return self.scenario
        return f"{self.scenario}/{self.visibility}/{self.weather}/{self.time}"


def _pick(dirnames, kind, camera_index):
    for pat in _CAM_NAMES.get(camera_index, ()):
        n = pat.format(k=kind)
        if n in dirnames:
            return n
    return None


def find_sequences(root, camera_index=0):
    """All folders under root holding an RGB camera folder and a matching SS camera folder."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        rgb, ss = _pick(dirnames, "RGB", camera_index), _pick(dirnames, "SS", camera_index)
        if rgb and ss:
            depth = _pick(dirnames, "Depth", camera_index)
            full = os.path.normpath(os.path.abspath(dirpath))
            generated = not rgb.endswith(f"_{camera_index}")
            if generated:  # DataAcquired/<timestamp>
                parts = [os.path.basename(full), "-", "-", "-"]
            else:  # .../Scenario_X/HV|LV/Weather/Time: the last four path parts
                parts = full.split(os.sep)[-4:]
                parts = ["?"] * (4 - len(parts)) + parts
            out.append(Sequence(dirpath, *parts, rgb_dir=os.path.join(dirpath, rgb), ss_dir=os.path.join(dirpath, ss),
                                depth_dir=os.path.join(dirpath, depth) if depth else "", generated=generated))
            dirnames[:] = []  # don't descend into camera folders
    return sorted(out, key=lambda s: s.path)


def _frame_files(cam_dir, prefer_bin=False):
    """{frame_index: path}. Dataset layout prefers converted PNGs; SynDRA-Gen prefers the raw .bin."""
    pngs, bins = {}, {}
    for dirpath, _, files in os.walk(cam_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            m = _IDX.search(stem)
            if not m:
                continue
            ext = ext.lower()
            if ext in (".png", ".jpg", ".jpeg"):
                pngs[int(m.group(1))] = os.path.join(dirpath, f)
            elif ext == ".bin":
                bins[int(m.group(1))] = os.path.join(dirpath, f)
    return (bins or pngs) if prefer_bin else (pngs or bins)


def frame_pairs(seq: Sequence, camera_index=0, stride=1, max_frames=None):
    """[(index, rgb_path, ss_path)] for frames that have both an RGB image and a label."""
    rgb = _frame_files(seq.rgb_dir or os.path.join(seq.path, f"RGBCamera_{camera_index}"), seq.generated)
    ss = _frame_files(seq.ss_dir or os.path.join(seq.path, f"SSCamera_{camera_index}"), seq.generated)
    idx = sorted(set(rgb) & set(ss))[::stride]
    if max_frames:
        idx = idx[:max_frames]
    return [(i, rgb[i], ss[i]) for i in idx]


def depth_files(seq: Sequence):
    """{frame_index: path} of the depth camera, or {} if the sequence has none."""
    return _frame_files(seq.depth_dir, True) if seq.depth_dir else {}


# --------------------------------------------------------------------------- reading
def infer_size(nbytes, channels=(3, 4), hint=None):
    """(width, height, channels) for a raw frame of nbytes, trying the hint resolution first."""
    for w, h in ([tuple(hint)] if hint else []) + _PRESETS:
        for c in channels:
            if nbytes == w * h * c:
                return w, h, c
    raise ValueError(f"{nbytes} bytes doesn't match a known resolution with {channels} byte(s) per pixel; "
                     "set camera.width / camera.height in the SynDRA config")


def read_rgb(path, hint=None):
    """BGR uint8 image. Raw .bin: W*H*3 (BGR, SynDRA-Gen) or W*H*4 (BGRA); hint = (width, height)."""
    if path.endswith(".bin"):
        data = np.fromfile(path, dtype=np.uint8)
        try:
            w, h, c = infer_size(data.size, (3, 4), hint)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
        return np.ascontiguousarray(data.reshape(h, w, c)[..., :3])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    return img


def read_labels(path, shape=None):
    """Label image as (H, W) integer ids or (H, W, 3) RGB colors. shape = (H, W) of the paired RGB frame."""
    if path.endswith(".bin"):
        data = np.fromfile(path, dtype=np.uint8)
        hint = (shape[1], shape[0]) if shape is not None else None
        try:
            w, h, c = infer_size(data.size, (1, 3, 4), hint)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
        if c == 1:
            return data.reshape(h, w)
        return np.ascontiguousarray(data.reshape(h, w, c)[..., 2::-1])  # BGR(A) -> RGB
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise RuntimeError(f"Could not read {path}")
    if lab.ndim == 3:
        return np.ascontiguousarray(lab[..., 2::-1]) if lab.shape[2] >= 3 else lab[..., 0]
    return lab


def read_depth(path, shape):
    """Raw 16-bit depth (H, W) uint16. SynDRA-Gen .bin: two bytes per pixel, high byte first. Units are
    not documented: syndra_eval.py reports the ratio to the gauge-derived range so they can be identified."""
    if path.endswith(".bin"):
        data = np.fromfile(path, dtype=np.uint8)
        h, w = shape[:2]
        if data.size != w * h * 2:
            raise ValueError(f"{path}: {data.size} bytes, expected {w * h * 2} for a {w}x{h} depth frame")
        raw = data.reshape(h, w, 2).astype(np.uint16)
        return (raw[..., 0] << 8) | raw[..., 1]
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"Could not read {path}")
    return d if d.ndim == 2 else d[..., 0]


def class_mask(labels, value):
    """Boolean mask for one label value ([R,G,B] or int)."""
    if value is None:
        raise ValueError("Label value not set in configs/syndra.yaml (run scripts/syndra_labels.py)")
    if isinstance(value, (list, tuple)):
        if labels.ndim != 3:
            raise ValueError("Color label given but the label image is single-channel")
        return np.all(labels == np.asarray(value, dtype=labels.dtype), axis=-1)
    if labels.ndim == 3:
        raise ValueError("Integer label given but the label image is color")
    return labels == value


def ego_region_mask(labels, cfg):
    m = np.zeros(labels.shape[:2], bool)
    for name in cfg["ego_region"]:
        m |= class_mask(labels, cfg["labels"][name])
    return m


# --------------------------------------------------------------------------- ground-truth rails
def _runs(row):
    """Centers and widths of True runs in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return (starts + ends - 1) / 2.0, ends - starts


def _match(row_mask, x_pred, tol, allow_contains=False):
    """Center of the rail run nearest x_pred in one mask row, or None if none is within tol."""
    centers, widths = _runs(row_mask)
    if not len(centers):
        return None
    i = int(np.argmin(np.abs(centers - x_pred)))
    if abs(centers[i] - x_pred) <= tol:
        return float(centers[i])
    if allow_contains and abs(centers[i] - x_pred) <= widths[i] / 2.0:
        return float(x_pred)  # inside a merged run (two rails drawn together at a switch)
    return None


def _extrapolate(hist, y, n=24):
    """Rail x at row y from a straight-line fit of its last n accepted points."""
    if len(hist) < 6:
        return hist[-1][1]
    pts = np.asarray(hist[-n:], float)
    k, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return float(k * y + c)


def gt_rails(rail_mask, min_sep_px=6.0, step=1, center_x=None, alt_mask=None, max_gap=80):
    """Left/right ego-rail centerlines from a rail label mask, traced from the bottom of the image up.

    Each rail is followed by straight-line extrapolation of its recent points, so the trace bridges
    rows where the label drops out (1 px rails far away) or where another track's rails are drawn over
    the ego rails at a switch. alt_mask (e.g. rails of other tracks) is consulted only where rail_mask
    has nothing at the expected position. With center_x, a bottom row showing more than two rails
    starts on the pair around the camera axis.

    Returns (left, right) as (N, 2) arrays of (x, y), bottom to top, for rows where both rails were found.
    """
    h = rail_mask.shape[0]
    hist_l, hist_r, left, right = [], [], [], []
    gap = 0
    for y in range(h - 1, -1, -step):
        if not hist_l:  # find the starting pair
            centers, _ = _runs(rail_mask[y])
            if len(centers) == 2:
                l, r = centers
            elif len(centers) > 2 and center_x is not None:
                mids = (centers[:-1] + centers[1:]) / 2.0
                i = int(np.argmin(np.abs(mids - center_x)))
                l, r = centers[i], centers[i + 1]
            else:
                continue
            if r - l < min_sep_px:
                continue
            hist_l.append((y, float(l)))
            hist_r.append((y, float(r)))
            left.append((l, y))
            right.append((r, y))
            continue
        pl, pr = _extrapolate(hist_l, y), _extrapolate(hist_r, y)
        sep = pr - pl
        if sep < min_sep_px:
            break
        tol = max(2.5, 0.15 * sep)
        xl = _match(rail_mask[y], pl, tol)
        xr = _match(rail_mask[y], pr, tol)
        if alt_mask is not None:
            if xl is None:
                xl = _match(alt_mask[y], pl, tol, allow_contains=True)
            if xr is None:
                xr = _match(alt_mask[y], pr, tol, allow_contains=True)
        if xl is not None:
            hist_l.append((y, xl))
        if xr is not None:
            hist_r.append((y, xr))
        if xl is not None and xr is not None and xr - xl >= min_sep_px:
            left.append((xl, y))
            right.append((xr, y))
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return np.array(left, float).reshape(-1, 2), np.array(right, float).reshape(-1, 2)


def rail_masks(labels, cfg):
    """(ego rail mask, all-rails mask or None) from the config's ego_rails / other_rails label names."""
    ego = class_mask(labels, cfg["labels"][cfg["ego_rails"]])
    alt = None
    for name in cfg.get("other_rails") or []:
        v = cfg["labels"].get(name)
        if v is not None:
            alt = (ego.copy() if alt is None else alt) | class_mask(labels, v)
    return ego, alt


# --------------------------------------------------------------------------- metrics helpers
def polygon_mask(left, right, shape):
    m = np.zeros(shape[:2], np.uint8)
    if len(left) and len(right):
        poly = np.vstack([left, right[::-1]]).round().astype(np.int32)
        cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def lateral_error(pred_left, pred_right, gt_left, gt_right):
    """Per-row |x_pred - x_gt| (px) for both rails at rows covered by both. Returns (y, err_l, err_r)."""
    if not (len(pred_left) and len(gt_left)):
        return np.empty(0), np.empty(0), np.empty(0)
    gy = gt_left[:, 1]
    order = np.argsort(gy)
    y = pred_left[:, 1]
    inside = (y >= gy.min()) & (y <= gy.max())
    y = y[inside]
    gl = np.interp(y, gy[order], gt_left[order, 0])
    gr = np.interp(y, gy[order], gt_right[order, 0])
    return y, np.abs(pred_left[inside, 0] - gl), np.abs(pred_right[inside, 0] - gr)
