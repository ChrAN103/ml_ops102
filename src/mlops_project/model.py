from torch import nn
import torch
from lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score


class Model(LightningModule):
    """LSTM-based model for fake news detection."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing integer vocabulary indices.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            Training loss.
        """
        texts, labels = batch
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.long()
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.
        """
        texts, labels = batch
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.long()
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, logger=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, logger=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.
        """
        texts, labels = batch
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.long()
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
