from torch.utils.data import Dataset

from mlops_project.data import NewsDataset


def test_news_dataset():
    """Test the NewsDataset class."""
    dataset = NewsDataset("data/raw")
    assert isinstance(dataset, Dataset)
