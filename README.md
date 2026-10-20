# rail-forward-cv

Forward-camera computer vision for the UTCRS Rail Robot. It finds the ego path (the rails the cart is on) with TEP-Net, then estimates how far ahead the path is clear by using the known rail gauge as a ruler. Future work includes managing distance and speed with the forward facing camera for implementing braking technology.

Pipeline: camera -> TEP-Net ego path -> gauge ranging -> on-screen HUD and CSV log.


## Getting started

```
bash scripts/setup.sh              # x86 workstation / L4 server
bash scripts/setup.sh --jetson     # Jetson Thor
source .venv/bin/activate
python scripts/run_forward.py --source third_party/tep/data/egopath.jpg --show-crop   # sanity check on a sample image
python scripts/run_forward.py --source 0                                               # live USB camera
```

Camera, mount, track and vehicle settings all live in configs/camera.yaml. The intrinsics and braking numbers in there are placeholders until brake technology is discussed.


## Scripts

scripts/setup.sh
One-time environment setup. Creates .venv, installs dependencies, downloads the pretrained weights, and runs the tests. With --jetson it reuses JetPack's PyTorch, TensorRT and OpenCV instead of installing them from pip.

scripts/download_weights.py
Downloads the TEP-Net pretrained models (egopath.zip from railenium.eu) into weights/, plus the RailSem19 ego-path annotations into data/. If the download is blocked, download the zip in a browser and pass it with --zip.                                                                                                                       

scripts/run_forward.py
The main program. Runs ego-path detection and ranging on a USB camera, video file, image, or GStreamer pipeline. Draws the path, distance ticks and a HUD that turns red when the clear path is shorter than the stopping distance. Can save annotated output (--save) and a per-frame CSV of range and latency (--log). Use --runtime tensorrt for the TensorRT engine, and --show-crop to see the crop region fed to the model. For footage from a camera we haven't calibrated (e.g. a downloaded cab ride), add --no-range to draw only the ego path; --start/--end pick a time window in seconds.

scripts/viewer.py
Tkinter GUI for live viewing. Shows the camera feed with buttons to pause, toggle inference on/off, switch models, take snapshots and record video to output/. Works as a plain camera viewer at minimum.

scripts/calibrate_intrinsics.py
Checkerboard camera calibration. First run with --capture to grab 20-40 checkerboard photos (space to save, q to quit), then run again on those images to solve for fx, fy, cx, cy and distortion and write them into configs/camera.yaml. Aim for under 0.5 px reprojection error. Redo it any time the zoom or focus ring moves. (lab camera has adjustable zoom, future camera might not)

scripts/export_onnx.py
Exports a model to ONNX, and with --build also builds a TensorRT FP16 engine using trtexec. Build the engine on the machine that will run it; Thor and L4 engines are not interchangeable.

scripts/verify_weights.py
Checks the downloaded TEP-Net weights: files present, strict loading into the configured architecture, parameter count against the paper, and a sanity run on TEP-Net's sample frame. Writes overlays to output/verify/ and prints PASS/FAIL per model.

scripts/benchmark.py
Measures model-only inference latency (mean, p50, p99, FPS) for a given model and runtime. Does not include camera capture, preprocessing or drawing.

scripts/syndra_labels.py
Finds the label colors in SynDRA's segmentation images (they aren't published) and saves a contact sheet so configs/syndra.yaml can be filled in. Suggests which color is the ego rails.

scripts/syndra_eval.py
Evaluates TEP-Net and the ranging on SynDRA against ground truth from the labels: ego-path IoU, rail position error, clear distance, recovered camera height, and range error at fixed distances. Writes per-frame and per-sequence CSVs plus overlay images to output/syndra/.


## Library code (railcv/)

egopath.py - TEP-Net inference wrapper for PyTorch or TensorRT, with fixed or automatic cropping.
ranging.py - Gauge-based distance estimate, camera height and horizon self-checks, and stopping distance.
camera.py - Opens USB cameras (MJPEG 1080p30), video files, images, or GStreamer pipelines.
overlay.py - Draws the ego path, distance ticks, crop box and HUD.
trt_runtime.py - Small TensorRT 10 engine runner (no pycuda needed).
syndra.py - SynDRA sequence discovery, frame reading, ground-truth ego region and rail centerlines.


## Other folders

configs/ - camera.yaml (the cart camera), syndra.yaml (SynDRA label colors and eval settings), camera_syndra.yaml (SynDRA camera for run_forward/viewer).
tests/ - ranging math checks and an end-to-end smoke test (run with python -m pytest tests).
third_party/tep/ - vendored TEP-Net code (MIT license), including train.py and eval.py for fine-tuning.
weights/ - downloaded models (not committed).
data/ - videos, calibration images, annotations (not committed).
output/ - annotated videos, snapshots, logs (not committed).


## Models

The default is chromatic-laughter-5 (EfficientNet-B3, regression). logical-tree-1 (ResNet-18, regression) is the fastest. Ranging needs a regression or classification model; the segmentation models only draw a mask.

The pretrained weights are CC BY-NC-SA 4.0 (non-commercial). Anything beyond that requires retraining on our own data (future work).


## Before trusting the distances

1. Calibrate the camera intrinsics with the lens locked at the cart setting. The fx in the config is a placeholder.
2. Set a fixed crop for the mount (mount.crop in the config) using --show-crop.
3. Measure end-to-end latency from camera to brake signal. USB capture usually costs more than inference.
4. Expect a domain gap. TEP-Net was trained on train-cab footage, so a low cart-mounted camera on US track will likely need fine-tuning on our own labeled footage.

Ranging assumes level track and no camera roll. Curves and grade changes will throw off far-range estimates, so validate against markers at known distances on real track.


## Testing in simulation with SynDRA

SynDRA (WACV 2025) is a photorealistic Unreal Engine 5 railway dataset with pixel labels that separate the ego ("main") track from other tracks. That gives exact ground truth for the ego path, so it tests the model and the ranging before the cart hardware is ready. License is CC BY-NC 4.0: research use only, cite the paper.

What it tests and what it doesn't: SynDRA's camera is 3.5 m up (train-cab height), 90 deg FOV, so it checks TEP-Net and the ranging math, not our low cart viewpoint. Synthetic images also differ from real ones, so treat accuracy numbers as a robustness check.

1. Download one sequence first (the full set is about 1.8 TB) plus Utils from https://syndra.retis.santannapisa.it/syndra.html (MEGA links; the MEGA desktop app or `mega-get` from MEGAcmd handle large folders). Start with Scenario 0, HV/Sunny/Morning, into data/syndra/.
2. Convert the .bin frames with SynDRA's Utils/RGB_bin_to_png.py and SS_bin_to_png.py. Any PNGs under RGBCamera_0 and SSCamera_0 are picked up and paired by frame number.
3. Find the label colors:
   ```
   python scripts/syndra_labels.py --seq data/syndra/Scenario_0/HV/Sunny/Morning
   ```
   Check output/syndra/labels/contact_sheet.jpg and copy the values for railraised-main, ballast-int-main and sleeper-int-main into configs/syndra.yaml.
4. Evaluate:
   ```
   python scripts/syndra_eval.py --root data/syndra --model chromatic-laughter-5
   python scripts/syndra_eval.py --root data/syndra --match LV          # fog/rain only
   ```
   Or just watch the model on a sequence (the HUD's fitted camera height should read about 3.5 m):
   ```
   python scripts/run_forward.py --config configs/camera_syndra.yaml --source data/syndra/Scenario_0/HV/Sunny/Morning/RGBCamera_0/<png folder>
   ```

Reading the results:
- iou: overlap with the ground-truth ego region. The ground truth includes the full rail width and the model's path runs between rail centerlines, so a perfect prediction scores slightly under 1.
- lat_err_m: how far the predicted rails are from the true rails, in metres at that distance.
- height_gt_m should be about 3.5. If it is consistently off, the gauge assumption is off; implied_gauge_m gives the value to put in configs/syndra.yaml.
- height_pred_m vs 3.5 and range_err@Z show how much model error turns into distance error.
