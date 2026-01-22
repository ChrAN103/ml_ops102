import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import typer
from loguru import logger

def plot_metrics(
    log_dir: Path = Path("logs/training"), 
    version: int = None,
    output_dir: Path = Path("reports/figures")
):
    """
    Plots training and validation metrics from PyTorch Lightning CSV logs.
    
    Args:
        log_dir: Base directory where logs are saved (default: logs/training)
        version: Specific version to plot. If None, picks the latest version.
        output_dir: Directory to save the figures.
    """
    if not log_dir.exists():
        logger.error(f"Log directory not found: {log_dir}")
        return

    # Find the specific version directory
    if version is None:
        # Find all version_N folders
        versions = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
        if not versions:
            logger.error(f"No version directories found in {log_dir}")
            return
        # sort by version number
        latest_version = sorted(versions, key=lambda p: int(p.name.split("_")[1]))[-1]
        log_path = latest_version
    else:
        log_path = log_dir / f"version_{version}"

    metrics_file = log_path / "metrics.csv"
    if not metrics_file.exists():
        logger.error(f"metrics.csv not found in {log_path}")
        return

    logger.info(f"Loading metrics from {metrics_file}")
    df = pd.read_csv(metrics_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Training & Validation Loss ---
    plt.figure(figsize=(10, 6))
    
    # Training loss is usually logged per step
    # We drop NaNs to get clean lines
    train_loss = df[['step', 'train_loss_step']].dropna()
    # val_loss = df[['epoch', 'val_loss']].dropna()
    
    # Add validation loss mapped to steps (approximate based on max step per epoch)
    # Or just plot on secondary x-axis (Epochs). 
    # Here we'll plot Training Loss by Step
    
    sns.lineplot(data=train_loss, x='step', y='train_loss_step', label='Train Loss (Step)', alpha=0.6)
    
    # Plot Val Loss as points (since it's per epoch)
    # We need to map epoch to step to overlay them, or just plot on separate axes.
    # Let's verify if 'step' is logged for validation rows
    val_loss_with_step = df[['step', 'val_loss']].dropna()
    if not val_loss_with_step.empty:
        sns.lineplot(data=val_loss_with_step, x='step', y='val_loss', label='Val Loss', marker='o', linewidth=2)
    
    plt.title("Training & Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_dir / "loss_curve.png")
    logger.success(f"Saved loss_curve.png to {output_dir}")
    plt.close()

    # --- Plot 2: Accuracy ---
    plt.figure(figsize=(10, 6))
    
    train_acc = df[['step', 'train_acc_step']].dropna()
    val_acc = df[['step', 'val_acc']].dropna()

    if not train_acc.empty:
        sns.lineplot(data=train_acc, x='step', y='train_acc_step', label='Train Acc', alpha=0.6)
    
    if not val_acc.empty:
        sns.lineplot(data=val_acc, x='step', y='val_acc', label='Val Acc', marker='o', linewidth=2)
        
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig(output_dir / "accuracy_curve.png")
    logger.success(f"Saved accuracy_curve.png to {output_dir}")
    plt.close()

if __name__ == "__main__":
    typer.run(plot_metrics)