"""Nexora AI core package."""

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

__version__ = "0.1.0"
