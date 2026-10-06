"""Run TEP-Net ego-path detection + gauge ranging on a camera, video or image.

Examples:
  python scripts/run_forward.py --source data/videos/ride.mp4 --save output/ride_out.mp4
  python scripts/run_forward.py --source 0                         # USB camera /dev/video0
  python scripts/run_forward.py --source 0 --runtime tensorrt      # Jetson Thor
  python scripts/run_forward.py --source frame.jpg --no-show --save output/frame_out.jpg
Keys: q / Esc quit, space pause.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml

from railcv.camera import IMAGE_EXT, frames
from railcv.egopath import MODELS, EgoPathModel
from railcv.overlay import draw_crop, draw_egopath, draw_hud, draw_ticks
from railcv.ranging import MPH, CameraModel, estimate_range, stopping_distance


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index, video/image path, or GStreamer pipeline")
    p.add_argument("--model", default="chromatic-laughter-5", help=f"one of: {', '.join(MODELS)}")
    p.add_argument("--config", default="configs/camera.yaml")
    p.add_argument("--crop", default=None, help="'auto', 'none', or x0,y0,x1,y1 (overrides config)")
    p.add_argument("--runtime", default="pytorch", choices=["pytorch", "tensorrt"])
    p.add_argument("--engine", default=None, help="TensorRT engine path (default <model>/best_fp16.engine)")
    p.add_argument("--device", default=None)
    p.add_argument("--save", default=None, help="write annotated video/image here")
    p.add_argument("--log", default=None, help="write per-frame CSV (range, latency) here")
    p.add_argument("--no-show", action="store_true", help="headless (no window)")
    p.add_argument("--show-crop", action="store_true")
    p.add_argument("--start", type=float, default=0, help="video start time (s)")
    return p.parse_args()


def main():
    a = parse()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    cam = CameraModel.from_yaml(a.config)
    veh = cfg["vehicle"]
    v = veh["max_speed_mph"] * MPH
    stop_m = stopping_distance(v, veh["decel_mps2"], veh["latency_s"])

    crop = a.crop if a.crop is not None else (cfg["mount"].get("crop") or "auto")
    if crop == "none":
        crop = None
    elif isinstance(crop, str) and crop != "auto":
        crop = tuple(int(c) for c in crop.split(","))

    model = EgoPathModel(a.model, crop=crop, device=a.device, runtime=a.runtime, engine=a.engine)
    print(f"Model {a.model} ({model.method}) on {model.device} [{a.runtime}]  crop={crop}")
    print(f"Stopping distance at {veh['max_speed_mph']} mph: {stop_m:.1f} m "
          f"(decel {veh['decel_mps2']} m/s^2, latency {veh['latency_s']} s)")

    is_image = os.path.splitext(a.source)[1].lower() in IMAGE_EXT
    c = cfg["camera"]
    src = frames(a.source, width=c["width"], height=c["height"], fps=c["fps"])
    if a.start and not is_image and not a.source.isdigit():
        cap = cv2.VideoCapture(a.source)
        fps_file = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        for _ in range(int(a.start * fps_file)):
            next(src, None)

    writer, logw, logf = None, None, None
    if a.log:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
        logf = open(a.log, "w", newline="")
        logw = csv.writer(logf)
        logw.writerow(["frame", "infer_ms", "loop_ms", "clear_m", "clear_err_m", "cam_height_m", "horizon_y", "valid_anchors"])

    t_prev, i, paused = time.perf_counter(), 0, False
    for frame, t_cap in src:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if is_image and crop == "auto":
            for _ in range(49):  # let the auto-crop converge on a single still (as upstream detect.py does)
                model(rgb)
        path = model(rgb)
        rng = estimate_range(path.left, path.right, cam, crop=path.crop) if path.has_rails else None

        now = time.perf_counter()
        loop_ms, t_prev = (now - t_prev) * 1e3, now

        vis = draw_egopath(frame, path)
        vis = draw_ticks(vis, path, rng)
        if a.show_crop:
            vis = draw_crop(vis, path.crop)

        clear = rng.clear_m if rng else None
        warn = clear is None or clear < stop_m
        hud = [
            f"Clear path: {clear:.1f} m (+/-{rng.clear_err_m:.1f}/px)" if clear else "Clear path: --",
            f"Stop dist @ {veh['max_speed_mph']} mph: {stop_m:.1f} m",
            f"Infer {path.infer_ms:.1f} ms | loop {loop_ms:.0f} ms ({1e3 / max(loop_ms, 1e-3):.0f} FPS)",
        ]
        if rng is not None and rng.cam_height_m:
            ref = f" (tape {cam.mount_height_m} m)" if cam.mount_height_m else ""
            hud.append(f"Fitted cam height {rng.cam_height_m:.2f} m{ref}")
        vis = draw_hud(vis, hud, warn=warn)

        if logw:
            logw.writerow([i, f"{path.infer_ms:.2f}", f"{loop_ms:.1f}",
                           f"{clear:.2f}" if clear else "",
                           f"{rng.clear_err_m:.2f}" if clear else "",
                           f"{rng.cam_height_m:.3f}" if rng and rng.cam_height_m else "",
                           f"{rng.horizon_y:.1f}" if rng and rng.horizon_y else "",
                           int(rng.valid.sum()) if rng else 0])

        if a.save:
            os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
            if is_image:
                cv2.imwrite(a.save, vis)
            else:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), c["fps"], (w, h))
                writer.write(vis)

        if not a.no_show:
            cv2.imshow("rail-forward-cv", vis)
            key = cv2.waitKey(0 if is_image else 1) & 0xFF
            if key == ord(" "):
                paused = not paused
            while paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    paused = False
                if key in (ord("q"), 27):
                    break
            if key in (ord("q"), 27):
                break
        i += 1

    if writer:
        writer.release()
    if logf:
        logf.close()
    if not a.no_show:
        cv2.destroyAllWindows()
    print(f"Processed {i} frame(s).")


if __name__ == "__main__":
    main()
