from pathlib import Path
import time

import hydra
import torch
from hydra.utils import to_absolute_path
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from mlops_project.model import Model
from mlops_project.data import NewsDataModule
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


class TorchProfilerCallback(Callback):
    """Torch.profiler integration.

    Enabled via Hydra config at `profiling.enabled=true`.
    Writes TensorBoard traces to: `<paths.log_dir>/<profiling.trace_subdir>`.
    """

    def __init__(self, profiling_cfg: DictConfig, log_dir: Path) -> None:
        super().__init__()
        self.cfg = profiling_cfg
        self.log_dir = log_dir
        self.prof = None
        self._entered = False

    def _activities(self):
        acts = [ProfilerActivity.CPU]

        use_cuda = getattr(self.cfg, "use_cuda", "auto")
        if use_cuda == "auto":
            if torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
        elif bool(use_cuda):
            if torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
            else:
                logger.warning("profiling.use_cuda=true but CUDA is not available; profiling CPU only.")

        return acts

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if not bool(getattr(self.cfg, "enabled", False)):
            return

        trace_dir = self.log_dir / str(getattr(self.cfg, "trace_subdir", "profile"))
        trace_dir.mkdir(parents=True, exist_ok=True)

        schedule_cfg = getattr(self.cfg, "schedule", None)
        if schedule_cfg is not None:
            sched = torch.profiler.schedule(
                wait=int(getattr(schedule_cfg, "wait", 1)),
                warmup=int(getattr(schedule_cfg, "warmup", 1)),
                active=int(getattr(schedule_cfg, "active", 3)),
                repeat=int(getattr(schedule_cfg, "repeat", 1)),
            )
        else:
            sched = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

        self.prof = profile(
            activities=self._activities(),
            schedule=sched,
            on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
            record_shapes=bool(getattr(self.cfg, "record_shapes", True)),
            profile_memory=bool(getattr(self.cfg, "profile_memory", False)),
            with_stack=bool(getattr(self.cfg, "with_stack", False)),
            with_flops=bool(getattr(self.cfg, "with_flops", False)),
            with_modules=bool(getattr(self.cfg, "with_modules", False)),
        )

        self.prof.__enter__()
        self._entered = True
        logger.info(f"Torch profiling enabled. Writing traces to: {trace_dir}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.prof is not None and self._entered:
            self.prof.step()

    def on_fit_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if self.prof is None or not self._entered:
            return

        try:
            sort_by = str(getattr(self.cfg, "table_sort_by", "self_cpu_time_total"))
            row_limit = int(getattr(self.cfg, "table_row_limit", 50))
            summary = self.prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)

            trace_dir = self.log_dir / str(getattr(self.cfg, "trace_subdir", "profile"))
            (trace_dir / "key_averages.txt").write_text(summary)
        except Exception as e:
            logger.warning(f"Failed to write profiler summary: {e}")
        finally:
            self.prof.__exit__(None, None, None)
            self._entered = False
            self.prof = None


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

    loggers = []
    csv_logger = CSVLogger(save_dir=log_dir, name="training")
    loggers.append(csv_logger)

    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            log_model=cfg.wandb.log_model,
            save_dir=log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)

    progress_bar = TQDMProgressBar(refresh_rate=1)

    callbacks = [checkpoint_callback, progress_bar]
    if getattr(cfg, "profiling", None) is not None and bool(getattr(cfg.profiling, "enabled", False)):
        callbacks.append(TorchProfilerCallback(cfg.profiling, log_dir=log_dir))

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )

    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    start_time = time.time()

    trainer.fit(model, data_module)

    end_time = time.time()
    training_time = end_time - start_time
    logger.success(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model saved to {best_model_path}")

    final_checkpoint_path = model_save_path / "model.pt"
    checkpoint_path = best_model_path if best_model_path else checkpoint_callback.last_model_path
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["vocab"] = data_module.vocab
        checkpoint["vocab_size"] = data_module.vocab_size
        torch.save(checkpoint, final_checkpoint_path)
        logger.info(f"Final model saved to {final_checkpoint_path}")

    if cfg.wandb.enabled:
        import wandb

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    train()
