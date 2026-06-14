import hashlib
import numpy as np
import onnxruntime as ort
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "v1": PROJECT_ROOT / "model" / "model_v1.onnx",
    "v2": PROJECT_ROOT / "model" / "model_v2.onnx",
}


sessions = {
    version: ort.InferenceSession(path)
    for version, path in MODEL_PATHS.items()
}


def choose_model_version(data):
    if "model_version" in data:
        return data["model_version"]

    client_id = data.get("client_id")

    if client_id is None:
        return "v1"

    hash_value = int(hashlib.md5(str(client_id).encode()).hexdigest(), 16)

    if hash_value % 2 == 0:
        return "v1"

    return "v2"


def get_predict(data):
    model_version = choose_model_version(data)

    if model_version not in sessions:
        raise ValueError(f"Unknown model_version: {model_version}")

    features = np.array(data["features"], dtype=np.float32)
    res = sessions[model_version].run(None, {"features": features})

    return res, model_version