#!/usr/bin/env bash
# Create .venv, install dependencies, download TEP-Net weights.
#   bash scripts/setup.sh            # x86 workstation / L4 server
#   bash scripts/setup.sh --jetson   # Jetson Thor: reuse JetPack's PyTorch, TensorRT and OpenCV
set -euo pipefail
cd "$(dirname "$0")/.."

JETSON=0
[[ "${1:-}" == "--jetson" ]] && JETSON=1

if [[ ! -d .venv ]]; then
  if [[ $JETSON == 1 ]]; then
    python3 -m venv --system-site-packages .venv   # see JetPack's torch/tensorrt/cv2
  else
    python3 -m venv .venv
  fi
fi
source .venv/bin/activate
python -m pip install -U pip wheel

if [[ $JETSON == 1 ]]; then
  python - <<'PY' || { echo "Install NVIDIA's PyTorch build for your JetPack first (or use NVIDIA's PyTorch container)."; exit 1; }
import torch, torchvision
assert torch.cuda.is_available(), "torch is installed but CUDA is not available"
print("torch", torch.__version__, "| cuda ok |", torch.cuda.get_device_name(0))
PY
  python -c "import tensorrt; print('tensorrt', tensorrt.__version__)" || echo "WARN: tensorrt python module not found (needed for --runtime tensorrt)"
  if python -c "import cv2" 2>/dev/null; then
    grep -v '^opencv-python' requirements.txt > /tmp/req_nocv.txt && pip install -r /tmp/req_nocv.txt
  else
    pip install -r requirements.txt
  fi
else
  python -c "import torch" 2>/dev/null || pip install torch torchvision
  pip install -r requirements.txt
fi

python scripts/download_weights.py || echo "WARN: weight download failed. Download https://tasv.railenium.eu/egopath.zip manually, then: python scripts/download_weights.py --zip <path>"
python -m pytest -q tests
echo "Setup complete. Activate with: source .venv/bin/activate"
