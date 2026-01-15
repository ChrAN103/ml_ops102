from pathlib import Path
import time

import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from mlops_project.model import Model
from mlops_project.data import NewsDataModule


def train(
    data_path: Path = Path("data/processed"),
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: Path = Path("models"),
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    accelerator: str = "auto",
    devices: int = 1,
) -> None:
    """Train the model using PyTorch Lightning."""
    data_module = NewsDataModule(data_path=data_path, batch_size=batch_size)
    data_module.setup("fit")

    model = Model(
        vocab_size=data_module.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    logger = CSVLogger(save_dir=model_save_path.parent / "logs", name="training")

    progress_bar = TQDMProgressBar(refresh_rate=1)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        enable_progress_bar=True,
    )

    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    trainer.fit(model, data_module)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved to {best_model_path}")

    final_checkpoint_path = model_save_path / "model.pt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = best_model_path if best_model_path else checkpoint_callback.last_model_path
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["vocab"] = data_module.vocab
        checkpoint["vocab_size"] = data_module.vocab_size
        torch.save(checkpoint, final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    typer.run(train)
