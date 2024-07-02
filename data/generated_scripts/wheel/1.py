from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
import torch

# A dataset with uniform weight distribution
class UniformDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = np.ones((length, size)) / (length * size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

# Split dataset into training and validating datasets
def generate_data(size, length):
    data = UniformDataset(size, length)
    train, val = torch.utils.data.random_split(data, [int(length * 0.8), length - int(length * 0.8)])
    train_loader = DataLoader(dataset=train, batch_size=100)
    val_loader = DataLoader(dataset=val, batch_size=100)
    return train_loader, val_loader

# Use PyTorch Lightning for constructing the neural network
class MechanicalAdvantageSystem(pl.LightningModule):
    def __init__(self, input_size):
        super(MechanicalAdvantageSystem, self).__init__()
        self.layer = torch.nn.Linear(input_size, input_size)
        
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

# Initialize the model and start training and validating
def train_model():
    train_loader, val_loader = generate_data(2, 1000)

    model = MechanicalAdvantageSystem(2)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model)

# Test with Mock
class TestMechanicalAdvantageSystem(TestCase):
    @patch('__main__.MechanicalAdvantageSystem', autospec=True)
    def test_mechanical_advantage_system(self, mock_model):
        instance = mock_model.return_value
        trainer.fit(instance)
        self.assertTrue(mock_model.called)
        mock_model.assert_called_once_with()

# Run the unit test
if __name__ == '__main__':
    train_model()
    tester = TestMechanicalAdvantageSystem()
    tester.test_mechanical_advantage_system()