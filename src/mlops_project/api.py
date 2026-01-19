import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import fastapi
import torch
from google.cloud import storage
from loguru import logger
from pydantic import BaseModel

from mlops_project.data import text_to_indices
from mlops_project.model import Model

GCS_BUCKET = os.getenv("GCS_BUCKET", "ml_ops_102_bucket")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "models/model.pt")
LOCAL_MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", "models/model.pt"))


class PredictionRequest(BaseModel):
    title: str
    text: str


ctx = {}


def download_model_from_gcs(bucket_name: str, source_blob: str, destination: Path) -> None:
    """Download model from Google Cloud Storage."""
    logger.info(f"Downloading model from gs://{bucket_name}/{source_blob}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination))
    logger.success(f"Model downloaded to {destination}")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    model_path = LOCAL_MODEL_PATH

    if not model_path.exists():
        download_model_from_gcs(GCS_BUCKET, GCS_MODEL_PATH, model_path)

    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    vocab = checkpoint.get("vocab")
    vocab_size = checkpoint.get("vocab_size")
    hyper_params = checkpoint.get("hyper_parameters", {})

    if hyper_params:
        hyper_params_clean = {k: v for k, v in hyper_params.items() if k != "vocab_size"}
        model = Model.load_from_checkpoint(
            model_path,
            vocab_size=vocab_size,
            **hyper_params_clean,
            strict=False,
        )
    else:
        model = Model.load_from_checkpoint(
            model_path,
            vocab_size=vocab_size,
            strict=False,
        )
    ctx["vocab"] = vocab
    ctx["model"] = model
    logger.success("Model loaded successfully")
    yield
    ctx.clear()
    logger.info("Model unloaded")


app = fastapi.FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: PredictionRequest) -> dict[str, bool | float]:
    if "model" not in ctx or "vocab" not in ctx:
        raise fastapi.HTTPException(status_code=500, detail="Model or vocabulary not loaded.")
    combined_text = request.title + " " + request.text
    combined_text = combined_text.lower().strip()
    input_indices = text_to_indices(combined_text, ctx["vocab"], max_length=200)
    ctx["model"].eval()
    with torch.no_grad():
        input_tensor = input_indices.unsqueeze(0)
        outputs = ctx["model"](input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()
        real = predicted_class == 1
    return {"prediction": real, "prob": predicted_prob}


@app.get("/")
def root():
    """Health check."""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
