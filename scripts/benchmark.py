"""Model-only latency benchmark (excludes camera, preprocessing and drawing).

  python scripts/benchmark.py --model chromatic-laughter-5
  python scripts/benchmark.py --model logical-tree-1 --runtime tensorrt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from railcv.egopath import EgoPathModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="chromatic-laughter-5")
    p.add_argument("--runtime", default="pytorch", choices=["pytorch", "tensorrt"])
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()

    m = EgoPathModel(a.model, crop=None, device=a.device, runtime=a.runtime)
    x = torch.rand(1, *m.config["input_shape"], device=m.device)
    for _ in range(20):
        m._infer(x)
    ts = []
    for _ in range(a.iters):
        if m.device.type == "cuda":
            torch.cuda.synchronize(m.device)
        t0 = time.perf_counter()
        m._infer(x)
        if m.device.type == "cuda":
            torch.cuda.synchronize(m.device)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    name = torch.cuda.get_device_name(m.device) if m.device.type == "cuda" else "cpu"
    print(f"{a.model} [{a.runtime}] on {name}: mean {ts.mean():.2f} ms | p50 {np.median(ts):.2f} | "
          f"p99 {np.percentile(ts, 99):.2f} | {1e3 / ts.mean():.0f} FPS")


if __name__ == "__main__":
    main()
