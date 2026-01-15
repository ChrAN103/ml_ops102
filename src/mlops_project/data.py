from pathlib import Path
from collections import Counter
import re

import pandas as pd
import torch
import typer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class NewsDataset(Dataset):
    """News dataset."""

    def __init__(self, data_path: Path, train: bool = True, vocab: dict[str, int] | None = None, max_length: int = 200) -> None:
        self.data_path = Path(data_path)
        self.train = train
        self.max_length = max_length
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        
        if self.data_path.is_dir():
            split_file = self.data_path / ("train.pt" if train else "test.pt")
        else:
            split_file = self.data_path.parent / ("train.pt" if train else "test.pt")
        
        if not split_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {split_file}. Run preprocessing first.")
        
        data = torch.load(split_file)
        self.texts = data["texts"]
        self.labels = data["labels"]
        
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
            if self.pad_token not in self.vocab:
                self.vocab[self.pad_token] = 0
            if self.unk_token not in self.vocab:
                self.vocab[self.unk_token] = 1
        
        self.vocab_size = len(self.vocab)

    def _build_vocab(self) -> dict[str, int]:
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in self.texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        vocab = {self.pad_token: 0, self.unk_token: 1}
        for word, count in word_counts.most_common():
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text to sequence of indices."""
        words = self._tokenize(text)
        unk_idx = self.vocab.get(self.unk_token, 1)
        indices = [self.vocab.get(word, unk_idx) for word in words]
        
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            pad_idx = self.vocab.get(self.pad_token, 0)
            indices = indices + [pad_idx] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)

    def _text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text to sequence of indices (internal method)."""
        return self.text_to_indices(text)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a given sample from the dataset."""
        text_indices = self._text_to_indices(self.texts[index])
        label = self.labels[index] if isinstance(self.labels[index], torch.Tensor) else torch.tensor(self.labels[index], dtype=torch.long)
        return text_indices, label

    def preprocess(self, output_folder: Path, test_size: float = 0.2) -> None:
        """Preprocess the raw data and save it to the output folder."""
        _preprocess_data(self.data_path, output_folder, test_size)


class NewsDataModule(LightningDataModule):
    """Lightning DataModule for News dataset."""

    def __init__(
        self,
        data_path: Path = Path("data/processed"),
        batch_size: int = 32,
        num_workers: int = 0,
        max_length: int = 200,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.vocab = None
        self.vocab_size = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = NewsDataset(self.data_path, train=True, max_length=self.max_length)
            self.vocab = self.train_dataset.vocab
            self.vocab_size = self.train_dataset.vocab_size
            
            self.val_dataset = NewsDataset(self.data_path, train=False, vocab=self.vocab, max_length=self.max_length)
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                if self.vocab is None:
                    train_ds = NewsDataset(self.data_path, train=True, max_length=self.max_length)
                    self.vocab = train_ds.vocab
                    self.vocab_size = train_ds.vocab_size
                self.test_dataset = NewsDataset(self.data_path, train=False, vocab=self.vocab, max_length=self.max_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def _preprocess_data(data_path: Path, output_folder: Path, test_size: float = 0.2) -> None:
    """Preprocess the raw data and save it to the output folder."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    df = df.dropna(subset=["text", "class"])

    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["combined_text"] = df["combined_text"].str.lower().str.strip()

    texts = df["combined_text"].tolist()
    labels = df["class"].astype(int).tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

    output_folder.mkdir(parents=True, exist_ok=True)

    torch.save({"texts": train_texts, "labels": torch.tensor(train_labels)}, output_folder / "train.pt")
    torch.save({"texts": test_texts, "labels": torch.tensor(test_labels)}, output_folder / "test.pt")

    print(f"Saved processed data to {output_folder}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data and save it to the output folder."""
    print("Preprocessing data...")
    data_path = Path(data_path)
    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_path = csv_files[0]
        print(f"Found CSV file: {data_path}")
    
    _preprocess_data(data_path, output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
