from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from mlops_project.data import NewsDataset


DATA_PATH = Path("data/processed")
FIXTURES_PATH = Path("tests/fixtures")


def get_data_path() -> Path:
    """Return the path to use for tests: processed data or fixtures."""
    if (DATA_PATH / "train.pt").exists():
        return DATA_PATH
    if (FIXTURES_PATH / "train.pt").exists():
        return FIXTURES_PATH
    pytest.skip(
        "No test data available (run: uv run python -c 'from mlops_project.data import create_test_fixtures; create_test_fixtures()')"
    )


def test_news_train_dataset():
    """Test the NewsDataset class."""
    data_path = get_data_path()
    dataset = NewsDataset(data_path, split="train")
    assert isinstance(dataset, Dataset)


def test_news_val_dataset():
    """Test the NewsDataset class."""
    data_path = get_data_path()
    dataset = NewsDataset(data_path, split="val")
    assert isinstance(dataset, Dataset)


def test_news_test_dataset():
    """Test the NewsDataset class."""
    data_path = get_data_path()
    dataset = NewsDataset(data_path, split="test")
    assert isinstance(dataset, Dataset)


def test_vocab_size():
    """Test if the vocabulary size is greater than zero."""
    data_path = get_data_path()
    dataset = NewsDataset(data_path)
    assert dataset.vocab_size > 0


def test_dataset_length():
    """Test if the dataset length is greater than zero."""
    data_path = get_data_path()
    dataset = NewsDataset(data_path)
    assert len(dataset) > 0


def test_sample_output():
    """Test if a sample from the dataset returns the correct types."""
    data_path = get_data_path()
    dataset = NewsDataset(data_path)
    text, label = dataset[0]
    assert isinstance(text, torch.Tensor)
    assert isinstance(label, torch.Tensor)
