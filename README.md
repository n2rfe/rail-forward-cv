# rail-forward-cv

Forward-camera computer vision for the UTCRS Rail Robot: **ego-path detection** (TEP-Net) plus **gauge-based monocular ranging** for the clear-path distance the brake design needs.

```
camera ──► TEP-Net (ego-path rails, 64 anchors) ──► ranging (Z = fx·G / dx) ──► HUD / CSV log
                                                            │
                                         clear-path distance vs. stopping distance @ 20 mph
```

## Quick start

```bash
code rail-forward-cv.code-workspace      # open in VS Code, accept the recommended extensions
bash scripts/setup.sh                    # x86 + L4 box   (Jetson Thor: bash scripts/setup.sh --jetson)
source .venv/bin/activate
python scripts/run_forward.py --source third_party/tep/data/egopath.jpg --show-crop   # sanity check
python scripts/run_forward.py --source 0                                               # live USB camera
```

In VS Code everything is also under **Run and Debug** (F5) and **Terminal → Run Task**.

| Launch config (F5) | Does |
|---|---|
| Forward: live camera | camera → ego path + range HUD |
| Forward: live camera (TensorRT, Thor) | same, TensorRT engine, logs to `output/live_log.csv` |
| Forward: video/image file | annotate a recorded ride → `output/annotated.mp4` + CSV |
| Forward: sample image | runs on TEP-Net's sample frame |
| Benchmark model | model-only latency |

| Task | Does |
|---|---|
| Setup (x86 / Jetson) | venv, deps, weights, tests |
| Export ONNX + build TensorRT FP16 engine | `weights/<model>/best_fp16.engine` via `trtexec` |
| Calibrate: capture / solve | checkerboard → intrinsics in `configs/camera.yaml` |
| List cameras (v4l2) | find the ELP's `/dev/video*` and its MJPEG modes |
| Run tests | ranging math + pipeline smoke test |

## Layout

```
railcv/
  egopath.py       TEP-Net inference wrapper (PyTorch or TensorRT 10), fixed or auto crop
  trt_runtime.py   TensorRT 10 engine runner (torch buffers, no pycuda)
  ranging.py       gauge ranging, robust dx-vs-row fit, camera-height/horizon self-check, stopping distance
  camera.py        USB (MJPEG 1080p30), video, image, or GStreamer sources
  overlay.py       path, distance ticks, HUD
scripts/           run_forward, download_weights, export_onnx, calibrate_intrinsics, benchmark, setup.sh
configs/camera.yaml  intrinsics, mount, gauge, speed/decel/latency  ← edit this
third_party/tep/   vendored TEP-Net (MIT): model code, train.py, eval.py
weights/           pretrained models (downloaded; CC BY-NC-SA 4.0)
```

## Models

| Weights | Backbone / head | Paper latency (TRT, RTX 6000) | IoU |
|---|---|---|---|
| `chromatic-laughter-5` (default) | EfficientNet-B3 / regression | 2.08 ms | 0.9753 |
| `logical-tree-1` | ResNet-18 / regression | 0.60 ms | 0.9695 |
| `twinkling-rocket-21` | EfficientNet-B3 / segmentation | 2.83 ms | 0.9769 |
| `brilliant-horse-15` | ResNet-18 / segmentation | 1.37 ms | 0.9737 |

Ranging needs rail polylines, so use a **regression** (or classification) model. Segmentation models only draw the mask.
Weights are **non-commercial** (CC BY-NC-SA 4.0, inherited from RailSem19), fine for research; retrain on your own data for anything else.

## Ranging

At each anchor row the two ego rails sit at about the same depth, so **Z = fx · G / dx**, with G = rail center-to-center spacing (1.505 m for standard gauge; TEP-Net rails are centerlines). On level track dx is linear in the image row, so a robust line fit averages out pixel noise and also yields:

- **Fitted camera height** — compare with the tape-measured lens height (`mount.height_m`). A mismatch means fx, the gauge, or the track geometry assumption is off.
- **Horizon row** — should sit near `cy` if the camera is level.

Error per pixel of dx is **Z² / (fx·G)** (~0.75 m at 45 m with the placeholder 5 mm intrinsics). Assumptions: level track, no camera roll; curves and grade changes bias the far range. Validate against markers at known distances on real track.

The HUD turns red when the clear path is shorter than `v·t + v²/2a` from `vehicle:` in the config (20 mph, placeholder 2 m/s², 0.5 s).

## Before trusting the numbers

1. **Calibrate intrinsics** with the zoom/focus rings taped at the cart setting. The `fx` in the config is a placeholder.
2. **Set a fixed crop** for the mount (`mount.crop`) using `--show-crop`; auto-crop is for unknown footage.
3. **Measure end-to-end latency** (camera → brake signal). USB MJPEG capture usually dominates, not inference.
4. **Expect a domain gap**: TEP-Net is trained on train-cab views (RailSem19). A low cart-mounted camera on US freight track needs fine-tuning on your own labeled footage (`third_party/tep/train.py`).

## Jetson Thor notes

- Run `bash scripts/setup.sh --jetson`. It reuses JetPack's PyTorch, TensorRT and OpenCV (GStreamer support) instead of pip builds.
- Build engines **on the Thor** (`Export ONNX + build TensorRT FP16 engine` task); engines are not portable between the Thor and the L4.
- The vendored `Detector` TensorRT path uses the TensorRT 8 API (removed in 10); this project uses `railcv/trt_runtime.py` instead.
- For remote work: VS Code **Remote-SSH** into the Thor and open this folder there.
