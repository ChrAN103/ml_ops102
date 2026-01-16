import fastapi
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel

from mlops_project.data import text_to_indices
from mlops_project.model import Model


class PredictionRequest(BaseModel):
    title: str
    text: str


ctx = {}


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    model_path: Path = Path("models/model.pt")
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
    try:
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
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
