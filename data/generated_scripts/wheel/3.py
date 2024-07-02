import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from unittest import TestCase
from unittest.mock import patch

# Load mock data
class WheelDataset(Dataset):

    def __init__(self, size=10, length=100):
        self.len = length
        self.data = torch.rand(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# Define network architecture
class WheelModel(pl.LightningModule):
  
    def __init__(self):
        super(WheelModel, self).__init__()
        self.layer = torch.nn.Linear(10, 3) # Let's assume the output tensor is representing ['load bearing', 'steering', 'equal weight distribution']
    
    # Forward pass
    def forward(self, x):
        return self.layer(x)

    # Criterion for loss calculation 
    def compute_loss(self, batch):
        return F.mse_loss(self.layer(batch), batch)

    # Training loop
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result
  
    # Validation loop
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result
  
    # Optimizer (Adam, in this case)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    # Loaders
    def train_dataloader(self):
        return DataLoader(dataset=WheelDataset(), batch_size=32)
  
    def test_dataloader(self):
        return DataLoader(dataset=WheelDataset(), batch_size=32)

# Test set up
class TestWheelModel(TestCase):

    @patch('__main__.WheelModel', autospec=True)
    def test_model(self, mock_model):
        instance = mock_model.return_value
        tr = pl.Trainer(max_epochs=2)
        tr.fit(instance)
        self.assertTrue(mock_model.called)
        mock_model.assert_called_once_with()

# Main method
if __name__ == "__main__":
    tr = pl.Trainer(max_epochs=2)
    model = WheelModel()
    tr.fit(model) # Training

    test_model = TestWheelModel() # Testing
    test_model.test_model()