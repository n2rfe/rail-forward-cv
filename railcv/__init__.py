"""Forward-camera ego-path detection and gauge-based ranging for the UTCRS rail cart."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEP_ROOT = os.path.join(ROOT, "third_party", "tep")
WEIGHTS_DIR = os.path.join(ROOT, "weights")

# The vendored TEP-Net code imports itself as `src.*`
if TEP_ROOT not in sys.path:
    sys.path.insert(0, TEP_ROOT)
