"""Verify the downloaded TEP-Net weights: files, loading, parameter count, and a sanity run.

  python scripts/verify_weights.py                 # every model folder in weights/
  python scripts/verify_weights.py logical-tree-1  # one model

For each model:
  files      config.yaml and best.pt present
  load       the state dict loads with strict=True into the architecture in config.yaml
             (catches truncated/corrupted downloads and wrong files)
  params     parameter count vs the TEP-Net paper (regression models)
  inference  runs on TEP-Net's sample frame (third_party/tep/data/egopath.jpg) and checks the
             prediction looks like an ego path: found, rails don't cross, starts near the bottom,
             widens toward the camera (classification/regression); non-empty mask (segmentation)
Overlays are written to output/verify/<model>.jpg so you can eyeball them.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import yaml

from railcv import TEP_ROOT, WEIGHTS_DIR
from railcv.egopath import MODELS, EgoPathModel, build_model
from railcv.overlay import draw_egopath, draw_hud

# Parameter counts from the TEP-Net paper (Table I), regression head
PAPER_PARAMS = {"efficientnet-b3": 14.57e6, "resnet18": 15.64e6}
SAMPLE = os.path.join(TEP_ROOT, "data", "egopath.jpg")


def check(name_or_path, device, out_dir):
    res = {"model": os.path.basename(name_or_path.rstrip("/"))}
    d = name_or_path if os.path.isdir(name_or_path) else os.path.join(WEIGHTS_DIR, name_or_path)
    cfg_p, w_p = os.path.join(d, "config.yaml"), os.path.join(d, "best.pt")
    res["files"] = os.path.isfile(cfg_p) and os.path.isfile(w_p)
    if not res["files"]:
        res["note"] = f"missing config.yaml/best.pt in {d}"
        return res
    with open(cfg_p) as f:
        cfg = yaml.safe_load(f)
    res["arch"] = f"{cfg['backbone']} / {cfg['method']}"
    res["size_mb"] = os.path.getsize(w_p) / 1e6

    try:
        net = build_model(cfg)
        net.load_state_dict(torch.load(w_p, map_location="cpu"), strict=True)
        res["load"] = True
    except Exception as e:  # noqa: BLE001
        res["load"], res["note"] = False, f"load failed: {str(e).splitlines()[0][:120]}"
        return res
    n = sum(p.numel() for p in net.parameters())
    res["params_m"] = n / 1e6
    ref = PAPER_PARAMS.get(cfg["backbone"]) if cfg["method"] == "regression" else None
    res["params"] = None if ref is None else abs(n - ref) / ref < 0.01

    img = cv2.imread(SAMPLE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m = EgoPathModel(d, crop="auto", device=device)
    for _ in range(30):  # let the auto-crop converge, as upstream detect.py does on stills
        path = m(rgb)
    h = img.shape[0]
    if cfg["method"] == "segmentation":
        area = (np.array(path.mask) > 0).mean() if path.mask is not None else 0.0
        ok = 0.005 < area < 0.5
        res["inference"], res["note"] = ok, f"mask covers {area:.1%} of the frame"
    else:
        problems = []
        if not path.has_rails or len(path.left) < 16:
            problems.append(f"only {len(path.left)} rail points")
        else:
            if np.any(path.left[:, 0] >= path.right[:, 0]):
                problems.append("rails cross")
            if path.left[:, 1].max() < 0.85 * h:
                problems.append("path doesn't start near the bottom")
            gap = path.right[:, 0] - path.left[:, 0]
            if not gap[0] > gap[-1]:
                problems.append("rails don't converge with distance")
        res["inference"] = not problems
        res["note"] = "; ".join(problems) or f"{len(path.left)} rail points, near gap {gap[0]:.0f}px -> far {gap[-1]:.0f}px"
    res["infer_ms"] = path.infer_ms

    vis = draw_egopath(img, path)
    vis = draw_hud(vis, [f"{res['model']} ({res['arch']})", "PASS" if res["inference"] else "CHECK"],
                   warn=not res["inference"])
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{res['model']}.jpg"), vis)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help=f"model names or folders (default: all in weights/; known: {', '.join(MODELS)})")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="output/verify")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    models = a.models or sorted(
        x for x in os.listdir(WEIGHTS_DIR) if os.path.isfile(os.path.join(WEIGHTS_DIR, x, "config.yaml"))
    )
    if not models:
        raise SystemExit("No models in weights/. Run: python scripts/download_weights.py")

    rows = [check(m, device, a.out) for m in models]
    yn = {True: "ok", False: "FAIL", None: "-"}
    print(f"\n{'model':<24}{'arch':<32}{'MB':>7}{'load':>6}{'params':>16}{'infer':>7}{'ms':>8}  notes")
    for r in rows:
        prm = f"{r['params_m']:.2f}M {yn[r.get('params')]}" if "params_m" in r else "-"
        print(f"{r['model']:<24}{r.get('arch', '-'):<32}{r.get('size_mb', 0):>7.1f}{yn[r.get('load')]:>6}"
              f"{prm:>16}{yn[r.get('inference')]:>7}{r.get('infer_ms', 0):>8.1f}  {r.get('note', '')}")
    good = all(r.get("load") and r.get("inference") and r.get("params") is not False for r in rows)
    print(f"\nDevice: {device}. Overlays: {a.out}/")
    print("ALL WEIGHTS OK" if good else "Some checks failed: see notes, and look at the overlays")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
