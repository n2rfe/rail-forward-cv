"""OpenCV drawing for the ego path, distance ticks and a status HUD (BGR images)."""

import cv2
import numpy as np

GREEN = (80, 189, 0)
ORANGE = (35, 80, 240)  # UTRGV orange in BGR
RED = (40, 40, 220)
WHITE = (255, 255, 255)


def draw_egopath(img, path, opacity=0.45):
    vis = img.copy()
    if path.mask is not None:
        m = np.array(path.mask) > 0
        color = np.zeros_like(vis)
        color[m] = GREEN
        return cv2.addWeighted(vis, 1.0, color, opacity, 0)
    if not path.has_rails:
        return vis
    poly = np.vstack([path.left, path.right[::-1]]).round().astype(np.int32)
    layer = vis.copy()
    cv2.fillPoly(layer, [poly], GREEN)
    vis = cv2.addWeighted(layer, opacity, vis, 1 - opacity, 0)
    for rail in (path.left, path.right):
        cv2.polylines(vis, [rail.round().astype(np.int32)], False, GREEN, 2, cv2.LINE_AA)
    return vis


def draw_ticks(img, path, rng, distances=(5, 10, 20, 30, 40, 50, 75, 100)):
    """Horizontal ticks across the ego path at fixed ranges, from the dx-vs-y fit."""
    if rng is None or rng.fit is None or not path.has_rails:
        return img
    ys = path.left[:, 1]
    y_top, y_bot = ys.min(), ys.max()
    for z in distances:
        y = rng.row_at_z(z)
        if y is None or not (y_top <= y <= y_bot):
            continue
        xl = np.interp(y, path.left[::-1, 1], path.left[::-1, 0])
        xr = np.interp(y, path.right[::-1, 1], path.right[::-1, 0])
        p1, p2 = (int(xl), int(y)), (int(xr), int(y))
        cv2.line(img, p1, p2, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{z} m", (p2[0] + 6, p2[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return img


def draw_crop(img, crop):
    if crop is not None:
        cv2.rectangle(img, (int(crop[0]), int(crop[1])), (int(crop[2]), int(crop[3])), ORANGE, 1)
    return img


def draw_hud(img, lines, warn=False):
    pad, lh = 10, 24
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for i, t in enumerate(lines):
        color = RED if (warn and i == 0) else WHITE
        cv2.putText(img, t, (pad, pad + lh * i + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img
