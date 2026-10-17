"""Options for the native prefill paths."""

import json
from pathlib import Path

CONFIG = json.loads((Path(__file__).parent / "profile.json").read_text())["options"]
