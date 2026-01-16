from pathlib import Path
import time

import hydra
import torch
from hydra.utils import to_absolute_path
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from mlops_project.model import Model
from mlops_project.data import NewsDataModule


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train the model using PyTorch Lightning with Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all hyperparameters and settings.
    """
    data_path = Path(to_absolute_path(cfg.data.data_path))
    model_save_path = Path(to_absolute_path(cfg.paths.model_save_path))
    log_dir = Path(to_absolute_path(cfg.paths.log_dir))

    model_save_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_module = NewsDataModule(data_path=data_path, batch_size=cfg.data.batch_size)
    data_module.setup("fit")

    model = Model(
        vocab_size=data_module.vocab_size,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    logger = CSVLogger(save_dir=log_dir, name="training")

    progress_bar = TQDMProgressBar(refresh_rate=1)

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        enable_progress_bar=True,
    )

    print(f"Starting training for {cfg.training.epochs} epochs...")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    start_time = time.time()

    trainer.fit(model, data_module)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved to {best_model_path}")

    final_checkpoint_path = model_save_path / "model.pt"
    checkpoint_path = best_model_path if best_model_path else checkpoint_callback.last_model_path
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["vocab"] = data_module.vocab
        checkpoint["vocab_size"] = data_module.vocab_size
        torch.save(checkpoint, final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    train()
