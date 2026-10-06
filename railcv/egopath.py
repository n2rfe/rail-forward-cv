"""TEP-Net ego-path inference with PyTorch or TensorRT 10 (JetPack 7 / Jetson Thor).

Wraps the vendored TEP-Net code (third_party/tep). The vendored Detector's TensorRT
path uses the TensorRT 8 API, which was removed in TensorRT 10, so engines are built
with `trtexec` (see scripts/export_onnx.py) and run here with the TensorRT 10 tensor API.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import v2 as T

from . import WEIGHTS_DIR  # noqa: F401  (also sets sys.path for `src`)
from src.nn.model import ClassificationNet, RegressionNet, SegmentationNet
from src.utils.autocrop import Autocropper
from src.utils.common import to_scaled_tensor
from src.utils.postprocessing import (
    classifications_to_rails,
    regression_to_rails,
    scale_mask,
    scale_rails,
)

# Pretrained weights shipped by the TEP-Net authors (egopath.zip)
MODELS = {
    "chromatic-laughter-5": "EfficientNet-B3, regression (most accurate regression)",
    "logical-tree-1": "ResNet-18, regression (fastest)",
    "twinkling-rocket-21": "EfficientNet-B3, segmentation",
    "brilliant-horse-15": "ResNet-18, segmentation",
    "fortuitous-goat-12": "EfficientNet-B3, classification",
    "fortuitous-pig-8": "ResNet-18, classification",
}


@dataclass
class EgoPath:
    """One frame's result. Rails are (N, 2) arrays of absolute pixel (x, y), bottom to top."""

    left: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    right: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    mask: Image.Image | None = None  # segmentation models only
    crop: tuple | None = None  # crop actually used for this frame
    infer_ms: float = 0.0

    @property
    def has_rails(self) -> bool:
        return len(self.left) > 0 and len(self.right) > 0

    def as_list(self):
        """Format expected by the vendored draw_egopath()."""
        if self.mask is not None:
            return self.mask
        return [self.left.tolist(), self.right.tolist()]


def resolve_model_dir(model: str) -> str:
    path = model if os.path.isdir(model) else os.path.join(WEIGHTS_DIR, model)
    if not os.path.isfile(os.path.join(path, "config.yaml")):
        raise FileNotFoundError(
            f"No model at {path}. Run `python scripts/download_weights.py` "
            f"(available: {', '.join(MODELS)})"
        )
    return path


def build_model(config: dict) -> torch.nn.Module:
    m = config["method"]
    if m == "regression":
        return RegressionNet(
            backbone=config["backbone"],
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
        )
    if m == "classification":
        return ClassificationNet(
            backbone=config["backbone"],
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            classes=config["classes"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
        )
    if m == "segmentation":
        return SegmentationNet(
            backbone=config["backbone"],
            decoder_channels=tuple(config["decoder_channels"]),
        )
    raise ValueError(f"Unknown method {m}")


class EgoPathModel:
    def __init__(self, model="chromatic-laughter-5", crop="auto", device=None, runtime="pytorch", engine=None):
        """
        Args:
            model: weights folder name under weights/ or a path to one (config.yaml + best.pt).
            crop: "auto", None, or (x_left, y_top, x_right, y_bottom) inclusive pixel coords.
                  With a fixed cart-mounted camera, a fixed crop is recommended (paper sec. VII-A).
            device: "cuda", "cuda:0", "cpu", ... (default: cuda if available).
            runtime: "pytorch" or "tensorrt".
            engine: path to a .engine file for runtime="tensorrt" (default: <model>/best_fp16.engine).
        """
        self.model_dir = resolve_model_dir(model)
        with open(os.path.join(self.model_dir, "config.yaml")) as f:
            self.config = yaml.safe_load(f)
        self.method = self.config["method"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.runtime = runtime
        self.resize = T.Resize(self.config["input_shape"][1:][::-1])

        if crop == "auto":
            self.autocrop = Autocropper(self.config)
            self.fixed_crop = None
        else:
            self.autocrop = None
            self.fixed_crop = tuple(crop) if crop is not None else None

        if runtime == "pytorch":
            self.model = build_model(self.config).to(self.device).eval()
            state = torch.load(os.path.join(self.model_dir, "best.pt"), map_location=self.device)
            self.model.load_state_dict(state)
            self._infer = self._infer_torch
        elif runtime == "tensorrt":
            from .trt_runtime import TRTEngine

            engine = engine or os.path.join(self.model_dir, "best_fp16.engine")
            self.trt = TRTEngine(engine, device=self.device)
            self._infer = self.trt
        else:
            raise ValueError(runtime)

    # ------------------------------------------------------------------ inference
    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.resize(to_scaled_tensor(img).unsqueeze(0)).to(self.device).contiguous()

    @torch.inference_mode()
    def _infer_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def current_crop(self):
        return self.autocrop() if self.autocrop is not None else self.fixed_crop

    def __call__(self, img) -> EgoPath:
        """img: PIL.Image (RGB) or HxWx3 uint8 RGB numpy array."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        shape = img.size
        crop = self.current_crop()
        src = img.crop((crop[0], crop[1], crop[2] + 1, crop[3] + 1)) if crop else img

        x = self._preprocess(src)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        pred = self._infer(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        infer_ms = (time.perf_counter() - t0) * 1e3
        pred = pred.float().cpu().numpy()

        out = EgoPath(crop=crop, infer_ms=infer_ms)
        if self.method == "regression":
            traj = pred[:, :-1].reshape(2, self.config["anchors"])
            ylim = 1 / (1 + np.exp(-pred[:, -1].item()))
            rails = regression_to_rails(traj, ylim)
        elif self.method == "classification":
            clf = pred.reshape(2, self.config["anchors"], self.config["classes"] + 1).argmax(axis=2)
            rails = classifications_to_rails(clf, self.config["classes"])
        else:
            mask = (pred.squeeze() > 0).astype(np.uint8) * 255
            out.mask = scale_mask(Image.fromarray(mask), crop, shape)
            rails = None

        if rails is not None and rails.size:
            rails = scale_rails(rails.astype(np.float64), crop, shape)
            out.left, out.right = rails[0], rails[1]

        if self.autocrop is not None:
            self.autocrop.update(shape, out.as_list())
        return out
