from __future__ import annotations

import sys
from pathlib import Path


# Ensure top-level modules like `ltlm_lightning.py` stay importable under pytest's
# importlib mode and under uv's isolated test environment.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
