"""verify_weights.py and run_forward.py --no-range/--end, with a random model (no download needed)."""

import os
import subprocess
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from railcv.egopath import build_model

CFG = {"method": "regression", "backbone": "resnet18", "input_shape": [3, 512, 512], "anchors": 64,
       "pool_channels": 8, "fc_hidden_size": 2048, "crop_margin_sides": 0.1, "crop_margin_top": 0.1}


def make_model(d, corrupt=False):
    d.mkdir()
    with open(d / "config.yaml", "w") as f:
        yaml.safe_dump(CFG, f)
    torch.manual_seed(0)
    state = build_model(CFG).state_dict()
    if corrupt:
        state.pop(next(iter(state)))
    torch.save(state, d / "best.pt")
    return str(d)


def run(args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT, check=False)


def test_verify_weights_loads_and_checks_params(tmp_path):
    good = make_model(tmp_path / "rand")
    bad = make_model(tmp_path / "broken", corrupt=True)
    r = run([os.path.join(ROOT, "scripts", "verify_weights.py"), good, bad, "--device", "cpu",
             "--out", str(tmp_path / "v")])
    lines = {ln.split()[0]: ln for ln in r.stdout.splitlines() if ln.startswith(("rand", "broken"))}
    assert "15.64M ok" in lines["rand"]  # matches the paper's ResNet-18 regression count
    assert "load failed" in lines["broken"]
    assert r.returncode == 1  # random weights can't produce a sane ego path / broken file fails
    assert (tmp_path / "v" / "rand.jpg").exists()


def test_run_forward_video_no_range_end(tmp_path):
    vid = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for k in range(50):  # 2 s at 25 fps
        wr.write(np.full((360, 640, 3), k * 4, np.uint8))
    wr.release()
    out, log = tmp_path / "out.mp4", tmp_path / "log.csv"
    r = run([os.path.join(ROOT, "scripts", "run_forward.py"), "--source", str(vid), "--model",
             make_model(tmp_path / "m"), "--device", "cpu", "--no-show", "--no-range",
             "--start", "0.4", "--end", "1.2", "--save", str(out), "--log", str(log)])
    assert r.returncode == 0, r.stderr
    assert "Processed 20 frame(s)" in r.stdout  # (1.2 - 0.4) s * 25 fps
    cap = cv2.VideoCapture(str(out))
    assert round(cap.get(cv2.CAP_PROP_FPS)) == 25
    cap.release()
