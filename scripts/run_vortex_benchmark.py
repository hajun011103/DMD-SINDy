from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vortex_benchmark import run_benchmark


if __name__ == "__main__":
    run_benchmark()
