from flask import Flask, request, jsonify
from datetime import datetime, timezone
import model_handler
import json
import time
import uuid
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def log_event(data):
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.info("APP_LOGL::" + json.dumps(data, ensure_ascii=False, default=str))


@app.route("/")
def index():
    return "Test message. The server is running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def pred():
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    data = request.get_json()
    result, model_version = model_handler.get_predict(data)

    prediction = result[0].tolist()
    probability = result[1]

    log_event({
        "event": "prediction",
        "request_id": request_id,
        "endpoint": "/predict",
        "status_code": 200,
        "model_version": model_version,
        "input_rows": len(data.get("features", [])),
        "prediction": prediction,
        "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
    })

    return jsonify({
        "model_version": model_version,
        "prediction": prediction,
        "probability": probability,
        "request_id": request_id
    }), 200