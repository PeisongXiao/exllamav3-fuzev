"""Configuration and context shared by the prefill kernels."""

from contextvars import ContextVar
import json
from pathlib import Path

CONFIG = json.loads((Path(__file__).parent / "profile.json").read_text())["options"]
ORIGINALS = {}
STAGE_CONTEXT = ContextVar("exllamav3_optimized_qsa", default=None)
