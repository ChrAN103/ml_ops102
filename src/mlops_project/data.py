from pathlib import Path

import pandas as pd
import torch
import typer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    """News dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        """Return a given sample from the dataset."""
        return self.texts[index], self.labels[index]

    def preprocess(self, output_folder: Path, test_size: float = 0.2) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

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
    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_path = csv_files[0]
        print(f"Found CSV file: {data_path}")
    dataset = NewsDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
