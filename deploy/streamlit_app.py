"""Entry point for the public deployment (Streamlit Community Cloud).

Serves DistilBERT V2 with its frozen calibration and decision from
``reports/evaluation_scenario_v1``. Paths are resolved from the repository
root; any variable already set in the environment takes precedence.
Weights come from Git LFS and are verified against their manifest on load:
if the host only has LFS pointer files, the app stops with an error.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "reports/evaluation_scenario_v1/distilbert-v2"

os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
for name, value in {
    "RECLAMORUTA_ARTIFACT_DIR": "artifacts/distilbert-v2",
    "RECLAMORUTA_DECISION": f"{FROZEN}/decision.json",
    "RECLAMORUTA_CALIBRATION": f"{FROZEN}/calibration.json",
    "RECLAMORUTA_DEVICE": "cpu",
    "RECLAMORUTA_MAX_PER_MINUTE": "20",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(name, value)

from reclamoruta.app import main  # noqa: E402

main()
