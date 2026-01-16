# The following code should be credited to Nicki Skafte Detlefsen and his MLOps course at 02476 - DTU.
# It has been copied from the exercises provided in the course and modified to fit the current project structure.

import matplotlib.pyplot as plt
import torch
import typer
from mlops_project.model import Model
from mlops_project.data import NewsDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: Path = Path("models/model.pt"), figure_name: str = "embeddings.png") -> None:
    """Visualize model embeddings using t-SNE."""
    if not model_checkpoint.exists():
        print(f"Model checkpoint not found at {model_checkpoint}")
        return

    print(f"Loading model from {model_checkpoint}")
    checkpoint = torch.load(model_checkpoint, map_location=DEVICE, weights_only=False)

    vocab = checkpoint.get("vocab")
    if not vocab:
        raise ValueError("Vocabulary not found in checkpoint. Ensure training saves it.")

    hyper_params = checkpoint.get("hyper_parameters", {})
    if "vocab_size" in hyper_params:
        del hyper_params["vocab_size"]

    model = Model.load_from_checkpoint(model_checkpoint, vocab_size=len(vocab), **hyper_params)
    model.to(DEVICE)
    model.eval()

    # Replace the classification head with Identity to extract embeddings
    model.fc = torch.nn.Identity()

    print("Loading test data...")
    vals_path = Path("data/processed")

    try:
        test_dataset = NewsDataset(data_path=vals_path, split="test", vocab=vocab)
    except FileNotFoundError:
        print("Test data not found, trying validation data...")
        test_dataset = NewsDataset(data_path=vals_path, split="val", vocab=vocab)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    print("Generating embeddings...")
    embeddings_list, targets_list = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(DEVICE)

            features = model(inputs)

            embeddings_list.append(features.cpu())
            targets_list.append(labels)

    if not embeddings_list:
        print("No embeddings generated.")
        return

    embeddings = torch.cat(embeddings_list).numpy()
    targets = torch.cat(targets_list).numpy()

    print(f"Embeddings shape: {embeddings.shape}")

    if embeddings.shape[1] > 500:
        print("Reducing dimensionality with PCA...")
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    print("Plotting...")
    plt.figure(figsize=(10, 10))

    classes = np.unique(targets)
    for c in classes:
        mask = targets == c
        label = "Fake" if c == 1 else "Real"
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f"Class {c} ({label})", alpha=0.6)

    plt.title("News Classification Embeddings t-SNE")
    plt.legend()

    output_path = Path("reports/figures") / figure_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    typer.run(visualize)
