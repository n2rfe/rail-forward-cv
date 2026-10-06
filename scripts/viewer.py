"""Live camera viewer GUI with optional TEP-Net ego-path inference.

Usage:
  python scripts/viewer.py                          # camera /dev/video0, no model
  python scripts/viewer.py --model chromatic-laughter-5
  python scripts/viewer.py --source data/videos/ride.mp4
  python scripts/viewer.py --runtime tensorrt       # Jetson TRT

Controls (buttons + keyboard):
  Space / Pause button  – pause / resume
  I / Inference toggle  – enable / disable model overlay
  S / Snapshot          – save a JPEG to output/
  R / Record            – start / stop MP4 recording to output/
  Q / Esc               – quit
"""

import argparse
import os
import sys
import threading
import time
from queue import Empty, Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import yaml

from railcv.camera import frames
from railcv.overlay import draw_hud

# Inference-stack imports are deferred: they require torch / TensorRT which may
# not be installed in the base environment (project uses a venv for those).
_INFER_AVAILABLE = False
MODELS: dict = {}
try:
    from railcv.egopath import MODELS, EgoPathModel
    from railcv.overlay import draw_crop, draw_egopath, draw_ticks
    from railcv.ranging import MPH, CameraModel, estimate_range, stopping_distance
    _INFER_AVAILABLE = True
except Exception as _e:
    print(f"[viewer] inference stack not available ({_e}); camera-only mode.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0")
    p.add_argument("--model", default=None, help="start with this model loaded (default: none)")
    p.add_argument("--config", default="configs/camera.yaml")
    p.add_argument("--crop", default=None)
    p.add_argument("--runtime", default="pytorch", choices=["pytorch", "tensorrt"])
    p.add_argument("--engine", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--show-crop", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Background capture + inference thread
# ---------------------------------------------------------------------------

class CaptureThread(threading.Thread):
    """Grabs frames, optionally runs inference, pushes (vis_bgr, stats_dict) onto a queue."""

    def __init__(self, source, cfg, cam_model, veh_stop_m, crop, show_crop, device, runtime, engine):
        super().__init__(daemon=True)
        self.source = source
        self.cfg = cfg
        self.cam_model = cam_model
        self.veh_stop_m = veh_stop_m
        self.crop_arg = crop
        self.show_crop = show_crop
        self.device = device
        self.runtime = runtime
        self.engine = engine

        self.queue = Queue(maxsize=2)       # latest frame; cap at 2 to stay current
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()  # set = paused
        self._infer_lock = threading.Lock()

        self.infer_enabled = False
        self.model_name = None
        self._ego_model = None             # loaded lazily when infer is enabled
        self._load_requested = None        # name of model to load next tick

        self.stats = {}
        self.recording = False
        self._writer = None
        self._writer_lock = threading.Lock()

    # ---- public API (called from GUI thread) ------

    def stop(self):
        self._stop_evt.set()

    def pause(self, state: bool):
        if state:
            self._pause_evt.set()
        else:
            self._pause_evt.clear()

    def request_model(self, name):
        """Ask the capture thread to load a new model (None = unload)."""
        with self._infer_lock:
            self._load_requested = name

    def set_infer(self, enabled: bool):
        self.infer_enabled = enabled

    def start_recording(self, path, fps, size):
        with self._writer_lock:
            if self._writer is None:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(path, fourcc, fps, size)
                self.recording = True

    def stop_recording(self):
        with self._writer_lock:
            if self._writer:
                self._writer.release()
                self._writer = None
            self.recording = False

    def _maybe_load_model(self):
        with self._infer_lock:
            req = self._load_requested
            self._load_requested = None
        if req is False:               # explicit unload sentinel
            self._ego_model = None
            self.model_name = None
            return
        if req is None:
            return
        # Load requested model
        self._ego_model = None
        self.model_name = None
        if req:
            if not _INFER_AVAILABLE:
                print("[viewer] inference stack not available; cannot load model")
                return
            try:
                self._ego_model = EgoPathModel(
                    req, crop=self.crop_arg, device=self.device,
                    runtime=self.runtime, engine=self.engine
                )
                self.model_name = req
            except Exception as e:
                print(f"[viewer] model load error: {e}")

    def run(self):
        c = self.cfg["camera"]
        veh = self.cfg["vehicle"]
        src = frames(self.source, width=c["width"], height=c["height"], fps=c["fps"])
        t_prev = time.perf_counter()

        for frame, _t_cap in src:
            if self._stop_evt.is_set():
                break
            while self._pause_evt.is_set() and not self._stop_evt.is_set():
                time.sleep(0.05)

            self._maybe_load_model()

            vis = frame.copy()
            clear_m = None
            infer_ms = 0.0

            if self.infer_enabled and self._ego_model is not None and _INFER_AVAILABLE:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                path = self._ego_model(rgb)
                rng = estimate_range(path.left, path.right, self.cam_model, crop=path.crop) if path.has_rails else None
                vis = draw_egopath(frame, path)
                vis = draw_ticks(vis, path, rng)
                if self.show_crop:
                    vis = draw_crop(vis, path.crop)
                clear_m = rng.clear_m if rng else None
                infer_ms = path.infer_ms

            now = time.perf_counter()
            loop_ms = (now - t_prev) * 1e3
            t_prev = now
            fps_live = 1e3 / max(loop_ms, 1e-3)

            hud_lines = []
            warn = False
            if self.infer_enabled and self._ego_model is not None:
                if clear_m is not None:
                    hud_lines.append(f"Clear path: {clear_m:.1f} m")
                    warn = clear_m < self.veh_stop_m
                else:
                    hud_lines.append("Clear path: --")
                    warn = True
                hud_lines.append(f"Stop dist @ {veh['max_speed_mph']} mph: {self.veh_stop_m:.1f} m")
                hud_lines.append(f"Infer {infer_ms:.1f} ms | {fps_live:.0f} FPS")
            else:
                hud_lines.append(f"{fps_live:.0f} FPS  |  {c['width']}x{c['height']}")
                if self.infer_enabled:
                    hud_lines.append("Loading model...")

            from railcv.overlay import draw_hud
            vis = draw_hud(vis, hud_lines, warn=warn)

            with self._writer_lock:
                if self._writer:
                    self._writer.write(vis)

            self.stats = {
                "fps": fps_live,
                "infer_ms": infer_ms,
                "clear_m": clear_m,
                "warn": warn,
                "model": self.model_name,
            }

            # Drop stale frames rather than queuing up
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put(vis)

        self.stop_recording()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ViewerApp:
    DISPLAY_W = 960
    DISPLAY_H = 540

    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.paused = False
        self.recording = False

        with open(args.config) as f:
            self.cfg = yaml.safe_load(f)

        if _INFER_AVAILABLE:
            self.cam_model = CameraModel.from_yaml(args.config)
            veh = self.cfg["vehicle"]
            v = veh["max_speed_mph"] * MPH
            self.veh_stop_m = stopping_distance(v, veh["decel_mps2"], veh["latency_s"])
        else:
            self.cam_model = None
            self.veh_stop_m = None

        crop = args.crop if args.crop is not None else (self.cfg["mount"].get("crop") or "auto")
        if crop == "none":
            crop = None
        elif isinstance(crop, str) and crop != "auto":
            crop = tuple(int(c) for c in crop.split(","))

        self.capture = CaptureThread(
            source=args.source,
            cfg=self.cfg,
            cam_model=self.cam_model,
            veh_stop_m=self.veh_stop_m,
            crop=crop,
            show_crop=args.show_crop,
            device=args.device,
            runtime=args.runtime,
            engine=args.engine,
        )

        self._build_ui()
        root.bind("<space>", lambda e: self._toggle_pause())
        root.bind("<KeyPress-i>", lambda e: self._toggle_infer())
        root.bind("<KeyPress-I>", lambda e: self._toggle_infer())
        root.bind("<KeyPress-s>", lambda e: self._snapshot())
        root.bind("<KeyPress-S>", lambda e: self._snapshot())
        root.bind("<KeyPress-r>", lambda e: self._toggle_record())
        root.bind("<KeyPress-R>", lambda e: self._toggle_record())
        root.bind("<KeyPress-q>", lambda e: self._quit())
        root.bind("<Escape>", lambda e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        # Load initial model if specified
        if args.model:
            self.model_var.set(args.model)
            self._on_model_change()
            self._toggle_infer()  # enable by default if model given

        self.capture.start()
        self._poll()

    # ---- UI construction ----

    def _build_ui(self):
        self.root.title("rail-forward-cv  |  live viewer")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        # ---- top: video canvas ----
        self.canvas = tk.Canvas(
            self.root, width=self.DISPLAY_W, height=self.DISPLAY_H,
            bg="black", highlightthickness=0
        )
        self.canvas.pack(padx=0, pady=0)

        # ---- bottom: control bar ----
        bar = tk.Frame(self.root, bg="#222", pady=6)
        bar.pack(fill=tk.X, padx=0)

        btn_style = dict(bg="#333", fg="white", activebackground="#555",
                         activeforeground="white", relief=tk.FLAT,
                         font=("Helvetica", 11), padx=10, pady=4, cursor="hand2")

        self.pause_btn = tk.Button(bar, text="⏸  Pause", command=self._toggle_pause, **btn_style)
        self.pause_btn.pack(side=tk.LEFT, padx=(8, 4))

        infer_label = "⬤  Inference OFF" if _INFER_AVAILABLE else "⬤  Inference (needs venv)"
        self.infer_btn = tk.Button(bar, text=infer_label, command=self._toggle_infer,
                                   state=tk.NORMAL if _INFER_AVAILABLE else tk.DISABLED, **btn_style)
        self.infer_btn.pack(side=tk.LEFT, padx=4)

        self.snap_btn = tk.Button(bar, text="📷  Snapshot", command=self._snapshot, **btn_style)
        self.snap_btn.pack(side=tk.LEFT, padx=4)

        self.rec_btn = tk.Button(bar, text="⏺  Record", command=self._toggle_record, **btn_style)
        self.rec_btn.pack(side=tk.LEFT, padx=4)

        # Model selector (hidden when inference stack is not available)
        model_names = list(MODELS.keys()) or ["(no models)"]
        self.model_var = tk.StringVar(value=self.args.model or model_names[0])
        if _INFER_AVAILABLE:
            lbl = tk.Label(bar, text="Model:", bg="#222", fg="#aaa", font=("Helvetica", 10))
            lbl.pack(side=tk.LEFT, padx=(16, 2))
            om = tk.OptionMenu(bar, self.model_var, *model_names, command=lambda _: self._on_model_change())
            om.config(bg="#333", fg="white", activebackground="#555", relief=tk.FLAT,
                      font=("Helvetica", 10), highlightthickness=0, indicatoron=True)
            om["menu"].config(bg="#333", fg="white")
            om.pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(bar, textvariable=self.status_var, bg="#222", fg="#888",
                          font=("Helvetica", 10), anchor="e")
        status.pack(side=tk.RIGHT, padx=10)

        self._tk_image = None  # keep reference to prevent GC

    # ---- event handlers ----

    def _toggle_pause(self):
        self.paused = not self.paused
        self.capture.pause(self.paused)
        if self.paused:
            self.pause_btn.config(text="▶  Resume")
        else:
            self.pause_btn.config(text="⏸  Pause")

    def _toggle_infer(self):
        enabled = not self.capture.infer_enabled
        self.capture.set_infer(enabled)
        if enabled:
            self.infer_btn.config(text="⬤  Inference ON", fg="#7dff7d")
        else:
            self.infer_btn.config(text="⬤  Inference OFF", fg="white")

    def _on_model_change(self):
        name = self.model_var.get()
        self.capture.request_model(name)
        self.status_var.set(f"Loading {name}…")

    def _snapshot(self):
        try:
            frame = self.capture.queue.get_nowait()
        except Empty:
            return
        os.makedirs("output", exist_ok=True)
        fname = f"output/snap_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        self.status_var.set(f"Saved {fname}")

    def _toggle_record(self):
        if not self.recording:
            c = self.cfg["camera"]
            fname = f"output/rec_{int(time.time())}.mp4"
            self.capture.start_recording(fname, c["fps"], (c["width"], c["height"]))
            self.recording = True
            self.rec_btn.config(text="⏹  Stop Rec", fg="#ff6b6b")
            self.status_var.set(f"Recording → {fname}")
        else:
            self.capture.stop_recording()
            self.recording = False
            self.rec_btn.config(text="⏺  Record", fg="white")
            self.status_var.set("Recording stopped")

    def _quit(self):
        self.capture.stop()
        self.root.destroy()

    # ---- frame polling ----

    def _poll(self):
        try:
            frame_bgr = self.capture.queue.get_nowait()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((self.DISPLAY_W, self.DISPLAY_H), Image.BILINEAR)
            self._tk_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

            s = self.capture.stats
            if s.get("model"):
                self.status_var.set(f"{s['model']}  |  {s['fps']:.0f} FPS  |  infer {s['infer_ms']:.1f} ms")
            elif not self.capture.infer_enabled:
                self.status_var.set(f"{s.get('fps', 0):.0f} FPS")
        except Empty:
            pass

        if not self.capture._stop_evt.is_set():
            self.root.after(16, self._poll)   # ~60 Hz UI refresh


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse()
    root = tk.Tk()
    app = ViewerApp(root, args)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
