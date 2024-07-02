import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from unittest import TestCase
from unittest.mock import patch

# Dataset Preparation
class DataPreparation:
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# PyTorch Model
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss

    def compute_loss(self, batch):
        loss = torch.nn.functional.mse_loss(self(batch), batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

# DataSet Building
def build_datasets():
    data = DataPreparation(2, 1000)
    train, val = torch.utils.data.random_split(data, [800, 200])
    train_loader = DataLoader(dataset=train, batch_size=100)
    val_loader = DataLoader(dataset=val, batch_size=100)
    return train_loader, val_loader

# Training
def training(model, max_epochs=100):
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model)

# Test Class
class ModelTest(TestCase):
    @patch('__main__.Model', autospec=True)
    def test_model(self, mock_model):
        instance = mock_model.return_value
        training(instance)
        self.assertTrue(mock_model.called)
        mock_model.assert_called_once_with()

# Program Start
if __name__ == "__main__":
    train_loader, val_loader = build_datasets()
    model = Model()
    training(model)

    obj = ModelTest()
    obj.test_model()