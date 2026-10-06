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
The main program. Runs ego-path detection and ranging on a USB camera, video file, image, or GStreamer pipeline. Draws the path, distance ticks and a HUD that turns red when the clear path is shorter than the stopping distance. Can save annotated output (--save) and a per-frame CSV of range and latency (--log). Use --runtime tensorrt for the TensorRT engine, and --show-crop to see the crop region fed to the model.

scripts/viewer.py
Tkinter GUI for live viewing. Shows the camera feed with buttons to pause, toggle inference on/off, switch models, take snapshots and record video to output/. Works as a plain camera viewer at minimum.

scripts/calibrate_intrinsics.py
Checkerboard camera calibration. First run with --capture to grab 20-40 checkerboard photos (space to save, q to quit), then run again on those images to solve for fx, fy, cx, cy and distortion and write them into configs/camera.yaml. Aim for under 0.5 px reprojection error. Redo it any time the zoom or focus ring moves. (lab camera has adjustable zoom, future camera might not)

scripts/export_onnx.py
Exports a model to ONNX, and with --build also builds a TensorRT FP16 engine using trtexec. Build the engine on the machine that will run it; Thor and L4 engines are not interchangeable.

scripts/benchmark.py
Measures model-only inference latency (mean, p50, p99, FPS) for a given model and runtime. Does not include camera capture, preprocessing or drawing.


## Library code (railcv/)

egopath.py - TEP-Net inference wrapper for PyTorch or TensorRT, with fixed or automatic cropping.
ranging.py - Gauge-based distance estimate, camera height and horizon self-checks, and stopping distance.
camera.py - Opens USB cameras (MJPEG 1080p30), video files, images, or GStreamer pipelines.
overlay.py - Draws the ego path, distance ticks, crop box and HUD.
trt_runtime.py - Small TensorRT 10 engine runner (no pycuda needed).


## Other folders

configs/ - camera.yaml, the one config file to edit.
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
