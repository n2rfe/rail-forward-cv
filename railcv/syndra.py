"""SynDRA dataset access and ground-truth extraction for ego-path / ranging evaluation.

Expected layout (from the SynDRA website):

    <root>/Scenario_0/HV/Sunny/Morning/RGBCamera_0/Bin_folder/Sce0_sun_mor_000001.bin
                                     /SSCamera_0/...
    <root>/Scenario_0/LV/Foggy/Afternoon/...

Run SynDRA's Utils/RGB_bin_to_png.py and SS_bin_to_png.py first; PNGs found anywhere under a
camera folder are used. Raw .bin frames are read only as a fallback (format guessed from size).
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


# --------------------------------------------------------------------------- config / camera
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def camera_model(cfg) -> CameraModel:
    """Pinhole model from the SynDRA sensor specs (90 deg HFOV, square pixels, no distortion)."""
    c = cfg["camera"]
    w, h = c["width"], c["height"]
    fx = (w / 2.0) / math.tan(math.radians(c["hfov_deg"]) / 2.0)
    return CameraModel(
        fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0,
        gauge_m=float(cfg["track"]["gauge_center_to_center_m"]),
        mount_height_m=float(c["height_m"]),
    )


# --------------------------------------------------------------------------- discovery
@dataclass
class Sequence:
    path: str
    scenario: str
    visibility: str  # HV / LV
    weather: str
    time: str

    @property
    def name(self):
        return f"{self.scenario}/{self.visibility}/{self.weather}/{self.time}"


def find_sequences(root, camera_index=0):
    """All folders under root that contain RGBCamera_<i> and SSCamera_<i>."""
    rgb, ss = f"RGBCamera_{camera_index}", f"SSCamera_{camera_index}"
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        if rgb in dirnames and ss in dirnames:
            # .../Scenario_X/HV|LV/Weather/Time: the last four path parts, whatever root was given
            parts = os.path.normpath(os.path.abspath(dirpath)).split(os.sep)[-4:]
            parts = ["?"] * (4 - len(parts)) + parts
            out.append(Sequence(dirpath, *parts))
            dirnames[:] = []  # don't descend into camera folders
    return sorted(out, key=lambda s: s.path)


def _frame_files(cam_dir):
    """{frame_index: path}, preferring converted PNGs over raw .bin."""
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
    return pngs or bins


def frame_pairs(seq: Sequence, camera_index=0, stride=1, max_frames=None):
    """[(index, rgb_path, ss_path)] for frames that have both an RGB image and a label."""
    rgb = _frame_files(os.path.join(seq.path, f"RGBCamera_{camera_index}"))
    ss = _frame_files(os.path.join(seq.path, f"SSCamera_{camera_index}"))
    idx = sorted(set(rgb) & set(ss))[::stride]
    if max_frames:
        idx = idx[:max_frames]
    return [(i, rgb[i], ss[i]) for i in idx]


# --------------------------------------------------------------------------- reading
def read_bin(path, width=1920, height=1080):
    """Best-effort raw reader (prefer SynDRA's own bin->png scripts). Returns BGR uint8.

    Guesses by file size: W*H*4 = BGRA8 (Unreal FColor), W*H*3 = BGR8, W*H*8 = RGBA float16.
    """
    data = np.fromfile(path, dtype=np.uint8)
    n = width * height
    if data.size == n * 4:
        return np.ascontiguousarray(data.reshape(height, width, 4)[..., :3])
    if data.size == n * 3:
        return data.reshape(height, width, 3)
    if data.size == n * 8:
        f = data.view(np.float16).reshape(height, width, 4)[..., :3].astype(np.float32)
        return np.clip(f[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(
        f"{path}: {data.size} bytes doesn't match {width}x{height} BGRA/BGR/RGBA16. "
        "Convert with SynDRA's Utils/*_bin_to_png.py instead."
    )


def read_rgb(path, width=1920, height=1080):
    """BGR uint8 image."""
    if path.endswith(".bin"):
        return read_bin(path, width, height)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    return img


def read_labels(path, width=1920, height=1080):
    """Label image as (H, W) int (single-channel PNG) or (H, W, 3) RGB uint8 (color PNG)."""
    if path.endswith(".bin"):
        return read_bin(path, width, height)[..., ::-1].copy()
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise RuntimeError(f"Could not read {path}")
    if lab.ndim == 3:
        return np.ascontiguousarray(lab[..., 2::-1]) if lab.shape[2] >= 3 else lab[..., 0]
    return lab


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


def gt_rails(rail_mask, min_sep_px=6.0, max_jump_px=None, step=1):
    """Left/right ego-rail centerlines from the railraised-main mask, tracked bottom-up.

    Returns (left, right) as (N, 2) arrays of (x, y), bottom to top, rows where both rails are seen.
    At switches the pair closest to the previous row's pair is kept.
    """
    h = rail_mask.shape[0]
    left, right = [], []
    prev = None
    for y in range(h - 1, -1, -step):
        centers, _ = _runs(rail_mask[y])
        if len(centers) < 2:
            if prev is not None and len(left) > 5:
                break  # path ended (occlusion / horizon) after it had started
            continue
        if prev is None:
            if len(centers) != 2:
                continue  # start only on an unambiguous row
            l, r = centers
        else:
            pl, pr = prev
            l = centers[np.argmin(np.abs(centers - pl))]
            r = centers[np.argmin(np.abs(centers - pr))]
            jump = max_jump_px or max(4.0, 0.25 * (pr - pl))
            if l == r or abs(l - pl) > jump or abs(r - pr) > jump:
                break
        if r - l < min_sep_px:
            break
        left.append((l, y))
        right.append((r, y))
        prev = (l, r)
    return np.array(left, float).reshape(-1, 2), np.array(right, float).reshape(-1, 2)


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
