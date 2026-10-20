"""Identify SynDRA label values so configs/syndra.yaml can be filled in.

SynDRA doesn't publish its label colors, so this samples a few label frames from one sequence,
lists every value with its pixel share, and saves a contact sheet showing where each value is.
It also suggests which value is the ego rails (railraised-main): thin, paired, bottom-center.

  python scripts/syndra_labels.py --seq data/syndra/Scenario_0/HV/Sunny/Morning
  -> output/syndra/labels/contact_sheet.jpg  (+ a YAML snippet printed to the terminal)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from railcv.syndra import _runs, find_sequences, frame_pairs, read_labels, read_rgb


def as_key(v):
    return tuple(int(x) for x in v) if np.ndim(v) else int(v)


def fmt(k):
    return f"[{k[0]}, {k[1]}, {k[2]}]" if isinstance(k, tuple) else str(k)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", required=True, help="a SynDRA sequence folder (or any folder above one)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--frames", type=int, default=6, help="label frames to sample")
    p.add_argument("--top", type=int, default=24, help="values to show")
    p.add_argument("--out", default="output/syndra/labels")
    a = p.parse_args()

    seqs = find_sequences(a.seq, a.camera)
    if not seqs:
        raise SystemExit(f"No RGBCamera_{a.camera}/SSCamera_{a.camera} folders under {a.seq}")
    seq = seqs[0]
    pairs = frame_pairs(seq, a.camera)
    if not pairs:
        raise SystemExit(f"No paired RGB/SS frames in {seq.path} (run SynDRA's bin->png scripts?)")
    pick = [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(a.frames, len(pairs))).astype(int)]
    print(f"Sequence {seq.name}: {len(pairs)} frames, sampling {len(pick)}")

    labs = [read_labels(s) for _, _, s in pick]
    rgbs = [read_rgb(r) for _, r, _ in pick]
    h, w = labs[0].shape[:2]
    color = labs[0].ndim == 3
    print(f"Label images: {w}x{h}, {'RGB color' if color else 'single-channel'}")

    # bottom-center zone where the ego track must be (camera is 0.3 m off the track centerline)
    zone = np.zeros((h, w), bool)
    zone[int(0.78 * h):, int(0.36 * w):int(0.64 * w)] = True

    stats = {}
    for lab in labs:
        flat = lab.reshape(-1, 3) if color else lab.reshape(-1)
        vals, inv, counts = np.unique(flat, axis=0 if color else None, return_inverse=True, return_counts=True)
        inv = inv.reshape(h, w)
        for i, (v, c) in enumerate(zip(vals, counts)):
            k = as_key(v)
            s = stats.setdefault(k, {"px": 0, "zone": 0, "runw": [], "pairs": 0, "rows": 0})
            m = inv == i
            s["px"] += int(c)
            s["zone"] += int((m & zone).sum())
            for y in range(int(0.80 * h), h, 8):  # rail signature: two thin runs per row near the bottom
                cx, cw = _runs(m[y, int(0.2 * w):int(0.8 * w)])
                if len(cx):
                    s["rows"] += 1
                    s["runw"].extend(cw.tolist())
                    s["pairs"] += int(len(cx) == 2)

    total = h * w * len(labs)
    ztotal = zone.sum() * len(labs)
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["px"])[: a.top]

    def rail_score(s):
        if s["rows"] < 3 or not s["runw"]:
            return 0.0
        thin = 1.0 / (1.0 + np.median(s["runw"]) / 15.0)
        return (s["zone"] / ztotal) * (s["pairs"] / s["rows"]) * thin

    rail_guess = max(stats.items(), key=lambda kv: rail_score(kv[1]))[0]

    print(f"\n{'#':>3}  {'value':<18} {'share':>7} {'in ego zone':>11} {'median run w':>12} {'2-run rows':>10}")
    for n, (k, s) in enumerate(ranked):
        mw = f"{np.median(s['runw']):.0f}px" if s["runw"] else "-"
        pr = f"{s['pairs'] / s['rows']:.0%}" if s["rows"] else "-"
        tag = "  <- rails? (railraised-main)" if k == rail_guess else ""
        print(f"{n:>3}  {fmt(k):<18} {s['px'] / total:>7.2%} {s['zone'] / ztotal:>11.1%} {mw:>12} {pr:>10}{tag}")

    # contact sheet: each value highlighted in magenta on the first sampled frame
    os.makedirs(a.out, exist_ok=True)
    tw, th = 480, 270
    cols = 4
    rows_n = (len(ranked) + cols - 1) // cols
    sheet = np.full((rows_n * (th + 28), cols * tw, 3), 30, np.uint8)
    lab0, rgb0 = labs[0], rgbs[0]
    for n, (k, s) in enumerate(ranked):
        m = np.all(lab0 == np.array(k), axis=-1) if color else lab0 == k
        vis = (rgb0 * 0.45).astype(np.uint8)
        vis[m] = (255, 0, 255)
        tile = cv2.resize(vis, (tw, th), interpolation=cv2.INTER_AREA)
        r, c = divmod(n, cols)
        y0, x0 = r * (th + 28), c * tw
        sheet[y0 + 28:y0 + 28 + th, x0:x0 + tw] = tile
        cv2.putText(sheet, f"#{n} {fmt(k)}  {s['px'] / total:.1%}", (x0 + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out = os.path.join(a.out, "contact_sheet.jpg")
    cv2.imwrite(out, sheet)
    cv2.imwrite(os.path.join(a.out, "sample_rgb.jpg"), rgb0)

    print(f"\nContact sheet: {out}")
    print("Find the ego-track classes in the sheet, then paste into configs/syndra.yaml -> labels:")
    print(f"  railraised-main: {fmt(rail_guess)}   # suggested: verify in the contact sheet")
    print("  ballast-int-main: ...   # between the ego rails")
    print("  sleeper-int-main: ...   # sleepers between the ego rails (may share a value with ballast)")


if __name__ == "__main__":
    main()
