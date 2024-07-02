import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from unittest import TestCase
from unittest.mock import patch, MagicMock

class WheelDataset(Dataset):
    def __init__(self, size=10, length=100):
        self.len = length
        self.data = torch.randn(length, size, dtype=torch.float)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class WheelModel(pl.LightningModule):
    def __init__(self):
        super(WheelModel, self).__init__()
        self.layer = torch.nn.Linear(10, 3)

    def forward(self, x):
        return self.layer(x)

    def compute_loss(self, batch):
        return F.mse_loss(self.layer(batch), batch)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss
  
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(dataset=WheelDataset(), batch_size=32)

    def val_dataloader(self):
        return DataLoader(dataset=WheelDataset(), batch_size=32)

# Mock test class to verify if the model runs and the methods get called correctly.
class TestWheelModel(TestCase):
    @patch('__main__.WheelModel.training_step', autospec=True)
    @patch('__main__.WheelModel.validation_step', autospec=True)
    @patch('__main__.WheelModel.configure_optimizers', autospec=True)
    def test_model(self, mock_training_step, mock_validation_step, mock_configure_optimizers):
        tr = pl.Trainer(max_epochs=2)
        model = WheelModel()
        tr.fit(model)

        mock_training_step.assert_called()
        mock_validation_step.assert_called()
        mock_configure_optimizers.assert_called()

if __name__ == "__main__":
    tr = pl.Trainer(max_epochs=2)
    model = WheelModel()
    tr.fit(model)

    test_model = TestWheelModel() 
    test_model.test_model()