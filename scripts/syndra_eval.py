"""Evaluate TEP-Net ego-path detection and gauge ranging on SynDRA.

For each frame: ground-truth ego path and rail centerlines come from the SynDRA labels, the model
predicts the ego path, and both rail sets are run through the same ranging code.

  python scripts/syndra_eval.py --root data/syndra --model chromatic-laughter-5
  python scripts/syndra_eval.py --root data/syndra/Scenario_0/HV/Sunny/Morning --max-frames 100 --vis 20

Outputs (output/syndra/<run>/): frames.csv (per frame), summary.csv (per sequence + overall),
vis/*.jpg (green = prediction, red = ground truth). Fill in configs/syndra.yaml labels first
(scripts/syndra_labels.py).

Metrics:
  iou            ego-path polygon vs GT ego region (rails + between rails)
  lat_err_px/m   |predicted rail x - GT rail x| per row, median; metres use the GT range at that row
  clear_*_m      distance to the far end of the predicted / GT ego path (ranging fit)
  height_*_m     camera height recovered by the ranging fit (truth: 3.5 m)
  range_err@Z    |predicted range - GT range| at the row where GT range = Z
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from railcv.egopath import MODELS, EgoPathModel
from railcv.overlay import RED, draw_egopath, draw_hud
from railcv.ranging import estimate_range
from railcv.syndra import (
    camera_model,
    depth_files,
    class_mask,
    ego_region_mask,
    find_sequences,
    frame_pairs,
    gt_rails,
    iou,
    lateral_error,
    load_config,
    polygon_mask,
    read_depth,
    read_labels,
    read_rgb,
)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="SynDRA root, a scenario, or a single sequence folder")
    p.add_argument("--config", default="configs/syndra.yaml")
    p.add_argument("--model", default="chromatic-laughter-5", help=f"one of: {', '.join(MODELS)} or a path")
    p.add_argument("--runtime", default="pytorch", choices=["pytorch", "tensorrt"])
    p.add_argument("--device", default=None)
    p.add_argument("--crop", default="auto", help="'auto', 'none', or x0,y0,x1,y1")
    p.add_argument("--stride", type=int, default=None, help="evaluate every Nth frame (default: config)")
    p.add_argument("--max-frames", type=int, default=None, help="per sequence")
    p.add_argument("--match", default=None, help="only sequences whose name contains this, e.g. 'LV' or 'Scenario_1'")
    p.add_argument("--vis", type=int, default=10, help="save every Nth evaluated frame as a JPG (0 = none)")
    p.add_argument("--video", action="store_true", help="also write one annotated MP4 per sequence")
    p.add_argument("--out", default=None, help="output folder (default output/syndra/<model>_<time>)")
    return p.parse_args()


def fnum(x, nd=2):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def eval_frame(model, rgb_bgr, labels, cfg, cam, depth=None):
    ev = cfg["eval"]
    gl, gr = gt_rails(class_mask(labels, cfg["labels"][cfg["ego_rails"]]), min_sep_px=ev["min_rail_sep_px"],
                      center_x=cam.cx)
    # ego region from the labels when the taxonomy has "between the rails" classes, else the area between GT rails
    gt_region = ego_region_mask(labels, cfg) if region_from_labels(cfg) else polygon_mask(gl, gr, labels.shape)
    path = model(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))

    r = {"infer_ms": path.infer_ms, "detected": int(path.has_rails), "gt_rows": len(gl)}
    pred_mask = polygon_mask(path.left, path.right, labels.shape) if path.has_rails else np.zeros(labels.shape[:2], bool)
    r["iou"] = iou(pred_mask, gt_region)

    rg = estimate_range(gl, gr, cam) if len(gl) >= 2 else None
    rp = estimate_range(path.left, path.right, cam, crop=path.crop) if path.has_rails else None
    r["clear_gt_m"] = rg.clear_m if rg else None
    r["clear_pred_m"] = rp.clear_m if rp else None
    r["height_gt_m"] = rg.cam_height_m if rg else None
    r["height_pred_m"] = rp.cam_height_m if rp else None
    # gauge implied by GT rails if the configured camera height is taken as truth: G_true = G_assumed * H_true / H_fit
    r["implied_gauge_m"] = (cam.gauge_m * cam.mount_height_m / rg.cam_height_m
                            if rg and rg.cam_height_m and cam.mount_height_m else None)

    y, el, er = lateral_error(path.left, path.right, gl, gr) if path.has_rails else (np.empty(0),) * 3
    if len(y):
        err = np.concatenate([el, er])
        r["lat_err_px"] = float(np.median(err))
        if rg is not None and rg.fit is not None:
            z = np.concatenate([rg.z_at_row(y)] * 2)
            ok = np.isfinite(z)
            r["lat_err_m"] = float(np.median(err[ok] * z[ok] / cam.fx)) if ok.any() else None
    for zc in ev["range_checkpoints_m"]:
        key = f"range_err@{zc}"
        r[key] = None
        if rg is None or rg.fit is None or rp is None or rp.fit is None:
            continue
        yc = rg.row_at_z(zc)
        y_lo = max(rg.y[rg.valid].min(), rp.y[rp.valid].min()) if rg.valid.any() and rp.valid.any() else None
        if y_lo is None or yc < y_lo or yc > labels.shape[0]:
            continue  # checkpoint beyond what either path covers
        r[key] = abs(float(rp.z_at_row(yc)) - zc)
    if depth is not None and rg is not None and rg.fit is not None:
        depth_metrics(r, depth, gl, gr, rg, rp, cfg)
    return r, path, (gl, gr), gt_region, rg, rp


def region_from_labels(cfg):
    names = [n for n in cfg.get("ego_region") or [] if n != cfg["ego_rails"]]
    return any(cfg["labels"].get(n) is not None for n in names)


def depth_metrics(r, depth, gl, gr, rg, rp, cfg):
    """Compare gauge ranging with the rendered depth map, sampled on the GT rails.

    depth_units_per_m: raw depth units per metre of gauge-derived range (identifies the depth scale;
    ~100 means centimetres). With depth.units_per_m set in the config, also:
      gauge_vs_depth_pct   |gauge range from GT rails - depth| / depth, median  (tests the ranging principle)
      range_err_depth_m/pct  |range from predicted rails - depth|, median over rows  (model + ranging)
    """
    h, w = depth.shape[:2]
    y = np.clip(gl[:, 1].round().astype(int), 0, h - 1)
    d = []
    for rail in (gl, gr):
        x = np.clip(rail[:, 0].round().astype(int), 0, w - 1)
        d.append(depth[y, x].astype(np.float64))
    d_raw = np.minimum(*d)  # the nearer of the two samples: guards against a miss onto the background
    z_g = rg.z_at_row(gl[:, 1])
    ok = (d_raw > 0) & (d_raw < 65535) & np.isfinite(z_g) & (z_g > 3) & (z_g < 80)
    if ok.sum() < 5:
        return
    r["depth_units_per_m"] = float(np.median(d_raw[ok] / z_g[ok]))
    upm = (cfg.get("depth") or {}).get("units_per_m")
    if not upm:
        return
    z_d = d_raw / float(upm)
    r["gauge_vs_depth_pct"] = float(np.median(np.abs(z_g[ok] - z_d[ok]) / z_d[ok]) * 100)
    if rp is not None and rp.fit is not None:
        z_p = rp.z_at_row(gl[:, 1])
        m = ok & np.isfinite(z_p) & (gl[:, 1] >= rp.y[rp.valid].min())
        if m.sum() >= 5:
            r["range_err_depth_m"] = float(np.median(np.abs(z_p[m] - z_d[m])))
            r["range_err_depth_pct"] = float(np.median(np.abs(z_p[m] - z_d[m]) / z_d[m]) * 100)


def draw(rgb_bgr, path, gt, gt_region, r, true_h=None):
    vis = draw_egopath(rgb_bgr, path)
    cnts, _ = cv2.findContours(gt_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, RED, 2)
    for rail in gt:
        if len(rail) > 1:
            cv2.polylines(vis, [rail.round().astype(np.int32)], False, RED, 1, cv2.LINE_AA)
    hud = [
        f"IoU {fnum(r['iou'])}   lateral err {fnum(r.get('lat_err_px'), 1)} px / {fnum(r.get('lat_err_m'), 3)} m",
        f"Clear: pred {fnum(r['clear_pred_m'], 1)} m | GT {fnum(r['clear_gt_m'], 1)} m",
        f"Cam height: pred {fnum(r['height_pred_m'])} | GT {fnum(r['height_gt_m'])}"
        + (f" | true {true_h:.2f} m" if true_h else ""),
        "green = TEP-Net, red = SynDRA ground truth",
    ]
    return draw_hud(vis, hud, warn=not path.has_rails)


def summarize(rows, keys):
    out = {"frames": len(rows), "detect_rate": np.mean([r["detected"] for r in rows]) if rows else float("nan")}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])], float)
        if k.startswith(("range_err", "lat_err", "clear", "depth", "gauge_vs")):
            out[k] = float(np.median(v)) if v.size else float("nan")
        else:
            out[k] = float(np.mean(v)) if v.size else float("nan")
    return out


def main():
    a = parse()
    cfg = load_config(a.config)
    if cfg["labels"].get(cfg["ego_rails"]) is None:
        raise SystemExit(f"Set the label value for '{cfg['ego_rails']}' in {a.config}\n"
                         f"Run: python scripts/syndra_labels.py --seq <sequence folder>")
    cam = camera_model(cfg)
    hint = (cfg["camera"]["width"], cfg["camera"]["height"])
    ci = cfg["camera"]["index"]
    stride = a.stride or cfg["eval"]["stride"]
    crop = None if a.crop == "none" else ("auto" if a.crop == "auto" else tuple(int(c) for c in a.crop.split(",")))

    seqs = [s for s in find_sequences(a.root, ci) if not a.match or a.match in s.name]
    if not seqs:
        raise SystemExit(f"No SynDRA sequences under {a.root}")
    out = a.out or os.path.join("output", "syndra", f"{os.path.basename(a.model.rstrip('/'))}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)
    print(f"{len(seqs)} sequence(s), stride {stride}, camera {ci}, fx {cam.fx:.1f} px, gauge {cam.gauge_m} m -> {out}")

    keys = ["iou", "lat_err_px", "lat_err_m", "clear_pred_m", "clear_gt_m", "height_pred_m", "height_gt_m",
            "implied_gauge_m", "infer_ms"] + [f"range_err@{z}" for z in cfg["eval"]["range_checkpoints_m"]] + [
            "depth_units_per_m", "gauge_vs_depth_pct", "range_err_depth_m", "range_err_depth_pct"]
    fcsv = open(os.path.join(out, "frames.csv"), "w", newline="")
    fw = csv.DictWriter(fcsv, ["sequence", "frame", "detected", "gt_rows"] + keys)
    fw.writeheader()

    by_seq = defaultdict(list)
    for seq in seqs:
        pairs = frame_pairs(seq, ci, stride=stride, max_frames=a.max_frames)
        if not pairs:
            print(f"  {seq.name}: no paired frames, skipped")
            continue
        model = EgoPathModel(a.model, crop=crop, device=a.device, runtime=a.runtime)  # fresh auto-crop per sequence
        writer = None
        dfiles = depth_files(seq)
        for n, (idx, rp_, sp_) in enumerate(pairs):
            rgb = read_rgb(rp_, hint)
            lab = read_labels(sp_, rgb.shape)
            if n == 0:
                cam = camera_model(cfg, rgb.shape)  # match the actual frame size
                print(f"  {seq.name}: {rgb.shape[1]}x{rgb.shape[0]}, fx {cam.fx:.1f} px"
                      + (", depth camera found" if dfiles else ""))
            depth = read_depth(dfiles[idx], rgb.shape) if idx in dfiles else None
            r, path, gt, region, _, _ = eval_frame(model, rgb, lab, cfg, cam, depth)
            fw.writerow({"sequence": seq.name, "frame": idx, **{k: fnum(r.get(k), 4) if k in keys else r[k]
                         for k in ["detected", "gt_rows"] + keys}})
            by_seq[seq.name].append(r)
            want_vis = a.vis and n % a.vis == 0
            if want_vis or a.video:
                vis = draw(rgb, path, gt, region, r, cam.mount_height_m)
                if want_vis:
                    cv2.imwrite(os.path.join(out, "vis", f"{seq.name.replace('/', '_')}_{idx:06d}.jpg"), vis)
                if a.video:
                    if writer is None:
                        writer = cv2.VideoWriter(os.path.join(out, f"{seq.name.replace('/', '_')}.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 10 / stride, vis.shape[1::-1])
                    writer.write(vis)
            print(f"\r  {seq.name}: {n + 1}/{len(pairs)}  IoU {fnum(r['iou'])}", end="", flush=True)
        if writer:
            writer.release()
        print()
    fcsv.close()

    all_rows = [r for rows in by_seq.values() for r in rows]
    table = [{"sequence": s, **summarize(rows, keys)} for s, rows in by_seq.items()]
    table.append({"sequence": "ALL", **summarize(all_rows, keys)})
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["sequence", "frames", "detect_rate"] + keys)
        w.writeheader()
        for t in table:
            w.writerow({k: (fnum(v, 4) if isinstance(v, float) else v) for k, v in t.items()})

    cols = ["frames", "detect_rate", "iou", "lat_err_m", "clear_pred_m", "clear_gt_m", "height_pred_m", "height_gt_m"]
    print("\n" + f"{'sequence':<36}" + "".join(f"{c:>14}" for c in cols))
    for t in table:
        print(f"{t['sequence']:<36}" + "".join(f"{(fnum(t[c], 3) if isinstance(t[c], float) else t[c]):>14}" for c in cols))
    ig = table[-1]["implied_gauge_m"]
    print(f"\nImplied gauge from GT rails (if the configured camera height is exact): {fnum(ig, 3)} m "
          f"(config: {cam.gauge_m} m). Median range error at checkpoints: "
          + ", ".join(f"{z} m -> {fnum(table[-1][f'range_err@{z}'], 2)}" for z in cfg["eval"]["range_checkpoints_m"]))
    last = table[-1]
    if math.isfinite(last.get("depth_units_per_m", float("nan"))):
        print(f"Depth map: {fnum(last['depth_units_per_m'], 1)} raw units per metre of gauge range "
              f"(about 100 = centimetres). Put it in the config as depth.units_per_m to get depth-based errors.")
    if math.isfinite(last.get("gauge_vs_depth_pct", float("nan"))):
        print(f"Gauge ranging vs depth (GT rails): {fnum(last['gauge_vs_depth_pct'], 2)} %   "
              f"predicted rails vs depth: {fnum(last['range_err_depth_m'], 2)} m ({fnum(last['range_err_depth_pct'], 2)} %)")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
