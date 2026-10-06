"""Frame sources: USB/UVC camera index, video file, image, or a GStreamer pipeline string."""

import os
import time

import cv2

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def open_source(src, width=1920, height=1080, fps=30):
    """Returns a cv2.VideoCapture (or None for a still image).

    src: "0" / 0 -> /dev/video0, path to a video, or a GStreamer pipeline (contains '!').
    """
    if isinstance(src, str) and os.path.splitext(src)[1].lower() in IMAGE_EXT:
        return None
    if isinstance(src, str) and "!" in src:
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    elif isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2) if os.name == "posix" else cv2.VideoCapture(int(src))
        # ELP USB cams only reach 1080p30 in MJPEG; YUYV is far slower over USB 2.0
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low: always grab the newest frame
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {src!r}")
    return cap


def frames(src, **kw):
    """Yields (bgr_frame, capture_timestamp_s). Works for images, videos and cameras."""
    cap = open_source(src, **kw)
    if cap is None:
        img = cv2.imread(src)
        if img is None:
            raise RuntimeError(f"Could not read {src}")
        yield img, time.perf_counter()
        return
    try:
        while True:
            ok, frame = cap.read()
            t = time.perf_counter()
            if not ok:
                break
            yield frame, t
    finally:
        cap.release()
