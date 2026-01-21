import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import fastapi
import torch
from loguru import logger
from prometheus_client import Counter, Histogram, Summary, make_asgi_app
from pydantic import BaseModel

from mlops_project.data import text_to_indices
from mlops_project.model import Model

prediction_requests = Counter("prediction_requests_total", "Total number of prediction requests")
prediction_errors = Counter("prediction_errors_total", "Total number of prediction errors")
health_checks = Counter("health_checks_total", "Total number of health check requests")
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)
text_length_summary = Summary("input_text_length_chars", "Summary of input text lengths in characters")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class PredictionRequest(BaseModel):
    title: str
    text: str


ctx = {}


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Loading model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    vocab = checkpoint.get("vocab")
    vocab_size = checkpoint.get("vocab_size")
    hyper_params = checkpoint.get("hyper_parameters", {})

    if hyper_params:
        hyper_params_clean = {k: v for k, v in hyper_params.items() if k != "vocab_size"}
        model = Model.load_from_checkpoint(
            MODEL_PATH,
            vocab_size=vocab_size,
            map_location=DEVICE,
            **hyper_params_clean,
            strict=False,
        )
    else:
        model = Model.load_from_checkpoint(
            MODEL_PATH,
            vocab_size=vocab_size,
            map_location=DEVICE,
            strict=False,
        )
    model = model.to(DEVICE)
    ctx["vocab"] = vocab
    ctx["model"] = model
    logger.success(f"Model loaded successfully on {DEVICE}")
    yield
    ctx.clear()
    logger.info("Model unloaded")


app = fastapi.FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.post("/predict")
async def predict(request: PredictionRequest) -> dict[str, bool | float]:
    prediction_requests.inc()
    with prediction_latency.time():
        try:
            if "model" not in ctx or "vocab" not in ctx:
                raise fastapi.HTTPException(status_code=500, detail="Model or vocabulary not loaded.")
            combined_text = request.title + " " + request.text
            combined_text = combined_text.lower().strip()
            text_length_summary.observe(len(combined_text))
            input_indices = text_to_indices(combined_text, ctx["vocab"], max_length=200)
            ctx["model"].eval()
            with torch.no_grad():
                input_tensor = input_indices.unsqueeze(0).to(DEVICE)
                outputs = ctx["model"](input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predicted_prob = probabilities[0, predicted_class].item()
                real = predicted_class == 1
            return {"prediction": real, "prob": predicted_prob}
        except Exception as e:
            prediction_errors.inc()
            raise fastapi.HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def root():
    health_checks.inc()
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "device": str(DEVICE),
    }
