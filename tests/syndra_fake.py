"""Builds a tiny SynDRA-shaped dataset with exact geometry, for tests (no real SynDRA needed)."""

import os

import cv2
import numpy as np

W, H = 1920, 1080
FX = 960.0  # 90 deg HFOV
CX, CY = W / 2, H / 2
CAM_H = 3.5
GAUGE = 1.505  # rail center-to-center
RAIL_W = 0.07
CAM_OFFSET = 0.3  # left camera sits 0.3 m left of the track centerline

COLORS = {  # arbitrary RGB label colors for the fake set
    "background": (70, 130, 180),
    "railraised-main": (255, 0, 0),
    "ballast-int-main": (0, 255, 0),
    "sleeper-int-main": (0, 0, 255),
    "railraised-other": (255, 255, 0),
}


def project(x_m, z):
    return CX + FX * x_m / z, CY + FX * CAM_H / z


def render(z_min=4.5, z_max=120.0, other_track=True):
    """Returns (bgr_image, rgb_label) of straight level track seen from the SynDRA camera."""
    img = np.full((H, W, 3), (90, 110, 120), np.uint8)
    img[: int(CY)] = (200, 170, 140)  # sky
    lab = np.zeros((H, W, 3), np.uint8)
    lab[:] = COLORS["background"]
    zs = np.geomspace(z_min, z_max, 1500)

    def strip(x_center, width_m, color_lab, color_img):
        pts_l = [project(x_center - width_m / 2, z) for z in zs]
        pts_r = [project(x_center + width_m / 2, z) for z in zs]
        poly = np.array(pts_l + pts_r[::-1]).round().astype(np.int32)
        cv2.fillPoly(lab, [poly], color_lab)
        cv2.fillPoly(img, [poly], color_img)

    off = CAM_OFFSET  # track centerline at +0.3 m in camera coordinates
    strip(off, GAUGE - RAIL_W, COLORS["ballast-int-main"], (95, 95, 95))
    for z in np.arange(z_min, z_max, 0.6):  # sleepers
        a = project(off - GAUGE / 2, z)
        b = project(off + GAUGE / 2, z + 0.25)
        cv2.rectangle(lab, tuple(map(int, a)), tuple(map(int, b)), COLORS["sleeper-int-main"], -1)
        cv2.rectangle(img, tuple(map(int, a)), tuple(map(int, b)), (60, 75, 90), -1)
    for s in (-1, 1):
        strip(off + s * GAUGE / 2, RAIL_W, COLORS["railraised-main"], (180, 180, 185))
    if other_track:
        for s in (-1, 1):
            strip(off + 4.5 + s * GAUGE / 2, RAIL_W, COLORS["railraised-other"], (170, 170, 175))
    return img, lab


def true_rails(z_min=6.0, z_max=100.0, n=64):
    zs = np.linspace(z_min, z_max, n)
    left = np.array([project(CAM_OFFSET - GAUGE / 2, z) for z in zs])
    right = np.array([project(CAM_OFFSET + GAUGE / 2, z) for z in zs])
    return left, right, zs


def make_tree(root, n_frames=3, label_ext=".png"):
    """root/Scenario_0/HV/Sunny/Morning/{RGBCamera_0,SSCamera_0}/png/Sce0_sun_mor_00000k.png"""
    seq = os.path.join(root, "Scenario_0", "HV", "Sunny", "Morning")
    img, lab = render()
    for cam in ("RGBCamera_0", "SSCamera_0", "RGBCamera_1", "SSCamera_1"):
        d = os.path.join(seq, cam, "png")
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(seq, cam, "Poses"), exist_ok=True)
        for k in range(1, n_frames + 1):
            out = os.path.join(d, f"Sce0_sun_mor_{k:06d}{label_ext}")
            cv2.imwrite(out, img if cam.startswith("RGB") else lab[..., ::-1])
    return seq
