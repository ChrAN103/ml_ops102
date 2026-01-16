import fastapi
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from mlops_project.model import Model
from mlops_project.data import text_to_indices
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    title: str
    text: str

ctx = {}

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    model_path: Path = Path("models/model.pt")
    data_path: Path = Path("data/processed")

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
    ctx['vocab'] = vocab
    ctx['model'] = model
    yield

app = fastapi.FastAPI(lifespan=lifespan)
        
@app.post("/predict")
async def predict(request: PredictionRequest) -> dict[str, float]:
    try:
        if 'model' not in ctx or 'vocab' not in ctx:
            raise fastapi.HTTPException(status_code=500, detail="Model or vocabulary not loaded.")
        combined_text= request.title + " " + request.text
        combined_text = combined_text.lower().strip()
        input_indices = text_to_indices(combined_text, ctx['vocab'], max_length=200)
        ctx['model'].eval()
        with torch.no_grad():
            input_tensor = input_indices.unsqueeze(0)
            outputs = ctx['model'](input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0, predicted_class].item()
            real = predicted_class == 1
        return {"prediction": real, "prob": predicted_prob}
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail={'error': str(e)})
    
@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the News Classification API"}