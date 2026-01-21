import os
import io
import json
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.cloud import storage
import pandas as pd
from datetime import datetime, timezone
import torch
import numpy as np
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, Info, Summary, make_asgi_app
from pydantic import BaseModel
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from fastapi.responses import HTMLResponse

from mlops_project.data import text_to_indices

import onnxruntime as ort

GCS_BUCKET = os.getenv("GCS_BUCKET", "ml_ops_102_bucket")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "models/model_optimized.onnx")
GCS_VOCAB_PATH = os.getenv("GCS_VOCAB_PATH", "models/vocab.json")
LOCAL_MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", f"/gcs/{GCS_BUCKET}/{GCS_MODEL_PATH}"))
LOAD_VOCAB_PATH = Path(os.getenv("LOCAL_VOCAB_PATH", f"/gcs/{GCS_BUCKET}/{GCS_VOCAB_PATH}"))
prediction_requests = Counter("prediction_requests_total", "Total number of prediction requests")
prediction_errors = Counter("prediction_errors_total", "Total number of prediction errors")
health_checks = Counter("health_checks_total", "Total number of health check requests")
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)
text_length_summary = Summary("input_text_length_chars", "Summary of input text lengths in characters")
prediction_classes = Counter("prediction_classes_total", "Predictions by class", ["class_label"])
prediction_confidence = Histogram(
    "prediction_confidence",
    "Model confidence distribution",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
model_info = Info("model", "Model information")
requests_in_progress = Gauge("requests_in_progress", "Number of requests currently being processed")


BUCKET_NAME = "ml_ops_102_bucket"

def get_onnx_providers():
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available and torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers
 
DEVICE = get_onnx_providers()

class PredictionRequest(BaseModel):
    title: str
    text: str


ctx = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = LOCAL_MODEL_PATH
    vocab_path = LOAD_VOCAB_PATH


    logger.info(f"Loading model from {model_path}")
    model = ort.InferenceSession(str(model_path), providers=DEVICE)

    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    ctx["vocab"] = vocab
    ctx["model"] = model
    vocab_size = len(vocab)
    model_info.info({"device": str(DEVICE), "model_path": str(model_path), "vocab_size": str(vocab_size)})
    logger.success(f"Model loaded successfully on {DEVICE}")
    yield
    ctx.clear()
    logger.info("Model unloaded")


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


# Save results to GCP
def save_prediction_to_gcp(
    now: str,
    title: str,
    text: str,
    prediction: int,
    predict_prob: float,
) -> None:
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # Prepare prediction data
    data = {"timestamp": now, "title": title, "text": text, "prediction": prediction, "probability": predict_prob}
    # Use a sanitized timestamp for the filename (replace spaces and colons)
    safe_timestamp = now.replace(" ", "_").replace(":", "-")
    blob = bucket.blob(f"logs/prediction_{safe_timestamp}.json")
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    print("Prediction saved to GCP bucket.")


def load_data(n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from GCS logs and training data for analysis.

    Args:
        n: Maximum number of latest GCS log files to load.

    Returns:
        Tuple containing reference data and current data formatted for drift analysis.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    try:
        train_blob = bucket.blob("data/processed/train.pt")
        train_bytes = train_blob.download_as_bytes()
        train_data_dict = torch.load(io.BytesIO(train_bytes), weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load training data from GCS: {exc}") from exc

    reference_data = pd.DataFrame(
        {
            "text": train_data_dict["texts"],
            "prediction": train_data_dict["labels"].tolist(),
        }
    )

    gcs_rows: list[dict[str, object]] = []
    try:
        blobs = list(bucket.list_blobs(prefix="logs/"))
        # Filter to only include prediction files (not directories)
        prediction_blobs = [b for b in blobs if b.name.startswith("logs/prediction_") and not b.name.endswith("/")]
        blobs_sorted = sorted(prediction_blobs, key=lambda b: b.updated or datetime.min, reverse=True)
        for blob in blobs_sorted[:n]:
            if blob.size is not None and blob.size == 0:
                logger.warning(f"Skipping empty log blob: {blob.name}")
                continue
            raw_text = blob.download_as_text().strip()
            if not raw_text:
                logger.warning(f"Skipping empty log blob: {blob.name}")
                continue
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping non-JSON log blob: {blob.name}, error: {e}")
                continue
            gcs_rows.append(
                {
                    "Title": data.get("title", ""),
                    "Text": data.get("text", ""),
                    "Prediction": data.get("prediction"),
                }
            )
    except Exception as exc:
        logger.warning(f"Failed to load GCS logs: {exc}")

    current_data = pd.DataFrame(gcs_rows, columns=["Title", "Text", "Prediction"])

    if current_data.empty:
        formatted_current_data = pd.DataFrame({"text": [], "prediction": []})
        return reference_data, formatted_current_data

    for col in ["Title", "Text", "Prediction"]:
        if col not in current_data.columns:
            current_data[col] = pd.NA

    current_data["Prediction"] = pd.to_numeric(current_data["Prediction"], errors="coerce")
    current_data["text"] = current_data["Title"].fillna("") + " " + current_data["Text"].fillna("")

    formatted_current_data = pd.DataFrame(
        {
            "text": current_data["text"],
            "prediction": current_data["Prediction"],
        }
    )
    return reference_data, formatted_current_data


def run_analysis(reference_data: pd.DataFrame, formatted_current_data: pd.DataFrame):
    """Run the data drift analysis and generate the report."""
    # Create and run the data drift report
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=formatted_current_data)
    # save locally
    report.save_html("monitoring.html")
    # also save to GCS with a timestamped filename
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    html_content = report.get_html()
    # save timestamped report to GCS
    blob = bucket.blob(f"reports/data_drift_report_{now}.html")
    blob.upload_from_string(html_content, content_type="text/html")

    # save/update latest report to GCS
    latest_blob = bucket.blob("reports/data_drift_report_latest.html")
    latest_blob.upload_from_string(html_content, content_type="text/html")

    logger.info(f"Report saved to GCS: reports/data_drift_report_{now}.html")

    return html_content

@app.post("/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks) -> dict[str, bool | float]:
    prediction_requests.inc()
    requests_in_progress.inc()
    try:
        with prediction_latency.time():
            if "model" not in ctx or "vocab" not in ctx:
                raise HTTPException(status_code=500, detail="Model or vocabulary not loaded.")
            
            combined_text = (request.title + " " + request.text).lower().strip()
            text_length_summary.observe(len(combined_text))
            
            input_indices = text_to_indices(combined_text, ctx["vocab"], max_length=200)
            input_tensor = input_indices.numpy().astype("int64").reshape(1, -1)
            
            outputs = ctx["model"].run(None, {"input": input_tensor})
            logits = outputs[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            predicted_class = int(np.argmax(probs))
            predicted_prob = float(probs[predicted_class])
            real = (predicted_class == 1)
            
            prediction_classes.labels(class_label="real" if real else "fake").inc()
            prediction_confidence.observe(predicted_prob)

            now = datetime.now(tz=timezone.utc).isoformat()
            background_tasks.add_task(
                save_prediction_to_gcp, now, request.title, request.text, predicted_class, predicted_prob
            )
            return {"prediction": real, "prob": predicted_prob}
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        requests_in_progress.dec()
@app.get("/report")
async def get_report(n: int | None = None):
    """Generate and return the report."""
    reference_data, formatted_current_data = load_data(n)
    html_content = run_analysis(reference_data, formatted_current_data)
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/")
def root():
    health_checks.inc()
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "device": str(DEVICE),
    }
