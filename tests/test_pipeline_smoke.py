"""End-to-end smoke test with a randomly initialized model (no downloaded weights needed).

Checks that preprocessing, inference, post-processing, ranging and drawing run together.
The predicted path is meaningless; accuracy is checked with real weights, not here.
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from railcv import TEP_ROOT
from railcv.egopath import EgoPathModel, build_model
from railcv.overlay import draw_egopath, draw_hud, draw_ticks
from railcv.ranging import CameraModel, estimate_range

CFG = {
    "method": "regression",
    "backbone": "resnet18",
    "input_shape": [3, 512, 512],
    "anchors": 64,
    "pool_channels": 8,
    "fc_hidden_size": 2048,
    "crop_margin_sides": 0.1,
    "crop_margin_top": 0.1,
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("random-regression")
    with open(d / "config.yaml", "w") as f:
        yaml.safe_dump(CFG, f)
    torch.manual_seed(0)
    torch.save(build_model(CFG).state_dict(), d / "best.pt")
    return str(d)


@pytest.mark.parametrize("crop", [None, "auto", (400, 300, 1500, 1079)])
def test_pipeline_runs(model_dir, crop):
    from PIL import Image

    img = np.array(Image.open(os.path.join(TEP_ROOT, "data", "egopath.jpg")).convert("RGB"))
    m = EgoPathModel(model_dir, crop=crop, device="cpu")
    for _ in range(3):
        path = m(img)
    assert path.infer_ms > 0
    if path.has_rails:
        assert path.left.shape[1] == 2
        rng = estimate_range(path.left, path.right, CameraModel(1786, 1786, 960, 540), crop=path.crop)
        vis = draw_ticks(draw_egopath(img[..., ::-1].copy(), path), path, rng)
    else:
        vis = draw_egopath(img[..., ::-1].copy(), path)
    vis = draw_hud(vis, ["Clear path: --", "test"], warn=True)
    assert vis.shape == img.shape
