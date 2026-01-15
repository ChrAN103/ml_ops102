from lightning.pytorch import LightningModule
import torch
from torch import nn


class Model(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing integer vocabulary indices.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Training loss.
        """
        x, y = batch
        y_hat = self(x)
        if y.dim() > 1:
            y = y.squeeze()
        loss = self.criterion(y_hat, y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Validation loss.
        """
        x, y = batch
        y_hat = self(x)
        if y.dim() > 1:
            y = y.squeeze()
        loss = self.criterion(y_hat, y.long())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Test loss.
        """
        x, y = batch
        y_hat = self(x)
        if y.dim() > 1:
            y = y.squeeze()
        loss = self.criterion(y_hat, y.long())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
