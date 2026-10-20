"""SynDRA tooling tests on a fake SynDRA-shaped dataset with exact geometry."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import syndra_fake as F

from railcv.camera import frames
from railcv.egopath import build_model
from railcv.ranging import estimate_range
from railcv.syndra import (
    camera_model,
    class_mask,
    find_sequences,
    frame_pairs,
    gt_rails,
    iou,
    lateral_error,
    load_config,
    polygon_mask,
    read_bin,
    read_labels,
    read_rgb,
)


def cfg_with_fake_labels(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "syndra.yaml"))
    for k, v in F.COLORS.items():
        if k in cfg["labels"]:
            cfg["labels"][k] = list(v)
    p = tmp_path / "syndra.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg, str(p)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("syndra")
    F.make_tree(str(root))
    return str(root)


def test_camera_model_matches_specs():
    cam = camera_model(load_config(os.path.join(ROOT, "configs", "syndra.yaml")))
    assert cam.fx == pytest.approx(960.0) and cam.fy == cam.fx
    assert (cam.cx, cam.cy) == (960.0, 540.0) and cam.mount_height_m == 3.5


def test_discovery_and_pairs(tree):
    seqs = find_sequences(tree)
    assert len(seqs) == 1
    s = seqs[0]
    assert (s.scenario, s.visibility, s.weather, s.time) == ("Scenario_0", "HV", "Sunny", "Morning")
    pairs = frame_pairs(s, stride=2)
    assert [p[0] for p in pairs] == [1, 3]
    assert find_sequences(s.path)[0].path == s.path  # pointing at the sequence itself works too
    assert len(find_sequences(tree, camera_index=1)) == 1


def test_gt_rails_and_ranging_recover_geometry(tree):
    pairs = frame_pairs(find_sequences(tree)[0])
    lab = read_labels(pairs[0][2])
    assert lab.ndim == 3
    rails = class_mask(lab, list(F.COLORS["railraised-main"]))
    gl, gr = gt_rails(rails)
    assert len(gl) > 200
    tl, tr, _ = F.true_rails()
    _, el, er = lateral_error(tl, tr, gl, gr)
    assert np.median(np.concatenate([el, er])) < 1.0  # centerlines within a pixel

    cam = camera_model(load_config(os.path.join(ROOT, "configs", "syndra.yaml")))
    r = estimate_range(gl, gr, cam)
    assert r.cam_height_m == pytest.approx(F.CAM_H, rel=0.02)
    assert r.horizon_y == pytest.approx(F.CY, abs=3)
    assert r.z_at_row(F.CY + F.FX * F.CAM_H / 20.0) == pytest.approx(20.0, rel=0.02)


def test_gt_rails_ignore_neighbor_track(tree):
    lab = read_labels(frame_pairs(find_sequences(tree)[0])[0][2])
    _, gr = gt_rails(class_mask(lab, list(F.COLORS["railraised-main"])))
    other = class_mask(lab, list(F.COLORS["railraised-other"]))
    assert other.any()
    assert gr[:, 0].max() < np.flatnonzero(other[-1]).min()


def test_polygon_iou():
    tl, tr, _ = F.true_rails()
    a = polygon_mask(tl, tr, (F.H, F.W))
    assert iou(a, a) == 1.0
    b = polygon_mask(tl + [5, 0], tr + [5, 0], (F.H, F.W))
    assert 0.5 < iou(a, b) < 1.0


def test_read_bin_bgra_roundtrip(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (F.H, F.W, 3), dtype=np.uint8)
    bgra = np.dstack([img, np.full((F.H, F.W), 255, np.uint8)])
    p = tmp_path / "x_000001.bin"
    bgra.tofile(p)
    assert np.array_equal(read_bin(str(p)), img)
    assert np.array_equal(read_rgb(str(p)), img)
    (tmp_path / "bad.bin").write_bytes(b"123")
    with pytest.raises(ValueError):
        read_bin(str(tmp_path / "bad.bin"))


def test_image_folder_source(tree):
    d = os.path.join(find_sequences(tree)[0].path, "RGBCamera_0", "png")
    assert sum(1 for _ in frames(d)) == 3


def test_label_inspector_suggests_rails(tree, tmp_path):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "syndra_labels.py"), "--seq", tree,
         "--frames", "2", "--out", str(tmp_path / "labels")],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout
    r, g, b = F.COLORS["railraised-main"]
    assert f"railraised-main: [{r}, {g}, {b}]" in out
    assert os.path.exists(tmp_path / "labels" / "contact_sheet.jpg")


def test_eval_end_to_end_random_model(tree, tmp_path):
    _, cfg_path = cfg_with_fake_labels(tmp_path)
    mdir = tmp_path / "rand-model"
    mdir.mkdir()
    mcfg = {"method": "regression", "backbone": "resnet18", "input_shape": [3, 512, 512], "anchors": 64,
            "pool_channels": 8, "fc_hidden_size": 2048, "crop_margin_sides": 0.1, "crop_margin_top": 0.1}
    with open(mdir / "config.yaml", "w") as f:
        yaml.safe_dump(mcfg, f)
    torch.manual_seed(0)
    torch.save(build_model(mcfg).state_dict(), mdir / "best.pt")
    out = tmp_path / "run"
    res = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "syndra_eval.py"), "--root", tree, "--config", cfg_path,
         "--model", str(mdir), "--device", "cpu", "--stride", "1", "--vis", "1", "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert (out / "frames.csv").exists() and (out / "summary.csv").exists()
    assert len(list((out / "vis").glob("*.jpg"))) == 3
    summary = (out / "summary.csv").read_text()
    assert "ALL" in summary
    # GT-only quantities are model-independent: height from GT rails must be ~3.5 m
    import csv

    with open(out / "summary.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["sequence"] == "ALL")
    assert float(row["height_gt_m"]) == pytest.approx(3.5, rel=0.03)
    assert float(row["implied_gauge_m"]) == pytest.approx(F.GAUGE, rel=0.03)
