from mlops_project.model import Model
from mlops_project.data import NewsDataset
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(epochs: int = 10):
    dataset = NewsDataset("data/raw")
    model = Model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for texts, labels in dataset:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item()}")


if __name__ == "__main__":
    train()
