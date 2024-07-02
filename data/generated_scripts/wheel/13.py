import numpy as np
import torch
import pytorch_lightning as pl
import unittest
import wheel
from torch.utils.data import DataLoader, Dataset
from unittest import TestCase
from unittest.mock import patch

# Here we define a simple Dataset
class SimpleDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = np.ones((length, size)) * (1 / (length * size))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# This function is used to generate data and split it into training and validation datasets
def generate_data(size, length):
    data = SimpleDataset(size, length)
    train, val = torch.utils.data.random_split(data, [int(length * 0.8), length - int(length * 0.8)])
    train_loader = DataLoader(dataset=train, batch_size=100)
    val_loader = DataLoader(dataset=val, batch_size=100)
    return train_loader, val_loader

# Here we define the LightningModule, our model
class MechanicalModel(pl.LightningModule):
    def __init__(self, input_size):
        super(MechanicalModel, self).__init__()
        self.layer = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_nb):
        loss = self.compute_loss(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss = self.compute_loss(batch)
        return {'val_loss': loss}

    def compute_loss(self, batch):
        loss = torch.nn.functional.mse_loss(self(batch), batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

# Finally, we need a function to run our model
def train_model():
    train_loader, val_loader = generate_data(2, 1000)

    model = MechanicalModel(2)

    trainer = pl.Trainer(max_epochs=100)

    trainer.fit(model)

# We would use mock to test our model.
class TestMechanicalModel(TestCase):
    @patch('__main__.MechanicalModel', autospec=True)
    def test_model(self, mock_mechanical_model):
        instance = mock_mechanical_model.return_value
        trainer.fit(instance)

        self.assertTrue(mock_mechanical_model.called)
        mock_mechanical_model.assert_called_once_with()

if __name__ == "__main__":
    train_model()
    tester = TestMechanicalModel()
    tester.test_model()