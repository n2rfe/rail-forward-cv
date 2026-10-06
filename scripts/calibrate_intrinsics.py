"""Checkerboard intrinsic calibration -> writes fx, fy, cx, cy, dist into configs/camera.yaml.

1. Lock the ELP lens zoom and focus at the setting you'll use on the cart (tape the rings).
2. Capture 20-40 photos of a printed checkerboard, filling different parts of the frame and tilted:
     python scripts/calibrate_intrinsics.py --capture 0 --out data/calib     # space = save, q = done
3. Calibrate:
     python scripts/calibrate_intrinsics.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square 0.025
   (--cols/--rows count INNER corners; --square is the square size in meters.)
Aim for RMS reprojection error < 0.5 px. Any zoom/focus change invalidates the result.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import yaml


def capture(src, out, w, h):
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    n = len(glob.glob(os.path.join(out, "*.jpg")))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imshow("capture (space=save, q=quit)", cv2.resize(f, None, fx=0.5, fy=0.5))
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            cv2.imwrite(os.path.join(out, f"calib_{n:03d}.jpg"), f)
            n += 1
            print(f"saved {n}")
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def calibrate(pattern, cols, rows, square, config):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    obj, img, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for path in sorted(glob.glob(pattern)):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        size = g.shape[::-1]
        ok, corners = cv2.findChessboardCorners(g, (cols, rows))
        if not ok:
            print(f"  no board: {path}")
            continue
        obj.append(objp)
        img.append(cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), crit))
    if len(obj) < 8:
        raise SystemExit(f"Only {len(obj)} usable images; need at least 8 (20+ recommended).")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    print(f"Images used: {len(obj)}   RMS reprojection error: {rms:.3f} px")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}\ndist={dist.ravel().round(5).tolist()}")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["intrinsics"] = {
        "fx": round(float(K[0, 0]), 2), "fy": round(float(K[1, 1]), 2),
        "cx": round(float(K[0, 2]), 2), "cy": round(float(K[1, 2]), 2),
        "dist": [round(float(d), 6) for d in dist.ravel()[:5]],
        "rms_px": round(float(rms), 3),
        "image_size": list(size),
    }
    with open(config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Updated {config} (comments in the file are not preserved).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", default=None, help="camera index to capture calibration images from")
    p.add_argument("--out", default="data/calib")
    p.add_argument("--images", default="data/calib/*.jpg")
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square", type=float, default=0.025, help="square size (m)")
    p.add_argument("--config", default="configs/camera.yaml")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    a = p.parse_args()
    if a.capture is not None:
        capture(a.capture, a.out, a.width, a.height)
    else:
        calibrate(a.images, a.cols, a.rows, a.square, a.config)
