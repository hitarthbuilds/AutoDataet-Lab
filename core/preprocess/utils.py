import joblib
from pathlib import Path
import json

def save_pipeline(pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

def load_pipeline(path: str):
    return joblib.load(path)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
