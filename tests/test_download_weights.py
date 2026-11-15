"""download_weights.py --zip on a Mac-made archive (.DS_Store, __MACOSX), like the real egopath.zip."""

import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mac_zip_is_handled(tmp_path):
    z = tmp_path / "egopath.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("__MACOSX/egopath/weights/._model-a", "junk")
        zf.writestr("__MACOSX/egopath/weights/model-a/._best.pt", "junk")
        zf.writestr("egopath/.DS_Store", "junk")
        zf.writestr("egopath/weights/.DS_Store", "junk")
        for m in ("model-a", "model-b"):
            zf.writestr(f"egopath/weights/{m}/config.yaml", "method: regression\n")
            zf.writestr(f"egopath/weights/{m}/best.pt", "weights")
            zf.writestr(f"egopath/weights/{m}/.DS_Store", "junk")
        zf.writestr("egopath/rs19_egopath.json", "{}")
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "weights", "data", "output", ".venv", "__pycache__"))
    r = subprocess.run([sys.executable, str(repo / "scripts" / "download_weights.py"), "--zip", str(z)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(repo / "weights")) == ["model-a", "model-b"]
    assert sorted(os.listdir(repo / "weights" / "model-a")) == ["best.pt", "config.yaml"]
    assert (repo / "data" / "rs19_egopath.json").exists()
    # re-running over existing folders works too
    r = subprocess.run([sys.executable, str(repo / "scripts" / "download_weights.py"), "--zip", str(z)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
