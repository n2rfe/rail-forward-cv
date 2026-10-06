"""Export a TEP-Net model to ONNX and (optionally) build a TensorRT FP16 engine with trtexec.

  python scripts/export_onnx.py chromatic-laughter-5            # -> weights/<model>/model.onnx
  python scripts/export_onnx.py chromatic-laughter-5 --build    # + best_fp16.engine via trtexec

Build the engine ON the machine that will run it (Thor engines don't run on the L4, and vice versa).
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from railcv.egopath import build_model, resolve_model_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--build", action="store_true", help="also build a TensorRT FP16 engine with trtexec")
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()

    mdir = resolve_model_dir(a.model)
    with open(os.path.join(mdir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).eval()
    model.load_state_dict(torch.load(os.path.join(mdir, "best.pt"), map_location="cpu"))

    onnx_path = os.path.join(mdir, "model.onnx")
    dummy = torch.rand(1, *cfg["input_shape"])
    torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["pred"],
                      opset_version=a.opset, dynamo=False)
    print(f"ONNX: {onnx_path}")

    if a.build:
        trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
        if not os.path.exists(trtexec):
            raise SystemExit("trtexec not found (on Jetson: /usr/src/tensorrt/bin/trtexec)")
        engine = os.path.join(mdir, "best_fp16.engine")
        cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine}", "--fp16"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Engine: {engine}")


if __name__ == "__main__":
    main()
