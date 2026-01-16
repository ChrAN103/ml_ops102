from torch.utils.data import Dataset
from mlops_project.data import NewsDataset
import torch


def test_news_train_dataset():
    """Test the NewsDataset class."""
    dataset = NewsDataset("data/processed",split="train")
    assert isinstance(dataset, Dataset)

def test_news_val_dataset():
    """Test the NewsDataset class."""
    dataset = NewsDataset("data/processed",split="val")
    assert isinstance(dataset, Dataset)

def test_news_test_dataset():
    """Test the NewsDataset class."""
    dataset = NewsDataset("data/processed",split="test")
    assert isinstance(dataset, Dataset)

def test_vocab_size():
    """Test if the vocabulary size is greater than zero."""
    dataset = NewsDataset("data/processed")
    assert dataset.vocab_size > 0

def test_dataset_length():
    """Test if the dataset length is greater than zero."""
    dataset = NewsDataset("data/processed")
    assert len(dataset) > 0

def test_sample_output():
    """Test if a sample from the dataset returns the correct types."""
    dataset = NewsDataset("data/processed")
    text, label = dataset[0]
    assert isinstance(text, torch.Tensor)
    assert isinstance(label, torch.Tensor)

