"""Download the TEP-Net pretrained weights + RailSem19 ego-path annotations.

Source: https://tasv.railenium.eu/egopath.zip  (CC BY-NC-SA 4.0, non-commercial)
Puts model folders in weights/ and the annotation JSON in data/rs19_egopath.json.
If the download is blocked, fetch the zip in a browser and pass it: --zip ~/Downloads/egopath.zip
"""

import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://tasv.railenium.eu/egopath.zip"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=None, help="use an already-downloaded egopath.zip")
    a = p.parse_args()

    tmp = tempfile.mkdtemp()
    try:
        zpath = a.zip
        if zpath is None:
            zpath = os.path.join(tmp, "egopath.zip")
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)

        src_w = next((os.path.join(r, d) for r, ds, _ in os.walk(tmp) for d in ds if d == "weights"), None)
        if src_w is None:
            raise SystemExit("No weights/ folder found in the zip")
        dst_w = os.path.join(ROOT, "weights")
        os.makedirs(dst_w, exist_ok=True)
        for name in os.listdir(src_w):
            dst = os.path.join(dst_w, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(os.path.join(src_w, name), dst)
            print(f"  weights/{name}")

        ann = next((os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f == "rs19_egopath.json"), None)
        if ann:
            os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
            shutil.copy(ann, os.path.join(ROOT, "data", "rs19_egopath.json"))
            print("  data/rs19_egopath.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
