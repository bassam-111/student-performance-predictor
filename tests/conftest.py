import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The Flask app loads model/model.pkl at import time. Train it first if it
# is missing (model.pkl is generated, not committed) so the app tests can
# import the app module.
MODEL_PATH = ROOT / "model" / "model.pkl"
if not MODEL_PATH.exists():
    sys.path.insert(0, str(ROOT / "model"))
    from train_model import main as train_main

    train_main()
