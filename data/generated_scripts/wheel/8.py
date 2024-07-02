# Required Libraries are imported.
import torch
import numpy as np
import pytorch_lightning as pl
from unittest import TestCase
from unittest.mock import patch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class WheelDataset(Dataset):
    '''This class is used to create a dummy dataset using PyTorch's Dataset class.'''

    def __init__(self, size=10, length=100):
        '''Initialize the class with size and length of data.'''
        self.len = length
        self.data = torch.rand(length, size)

    def __getitem__(self, index):
        '''Get elements of dataset as per the index.'''
        return self.data[index]

    def __len__(self):
        '''Get the length of dataset.'''
        return self.len

class WheelModel(pl.LightningModule):
    '''This class is used to create a PyTorch model using PyTorch Lightning's LightningModule class.'''  

    def __init__(self):
        '''Initialize the class and layers of neural network.'''
        super(WheelModel, self).__init__()
        self.layer = torch.nn.Linear(10, 3) 
    
    def forward(self, x):
        '''Define the forward pass.'''
        return self.layer(x)

    def compute_loss(self, batch):
        '''Calculate the loss using Mean Square Error Loss function.'''
        return F.mse_loss(self.layer(batch), batch)

    def training_step(self, batch, batch_idx):
        '''Define the training steps.'''
        loss = self.compute_loss(batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result
  
    def validation_step(self, batch, batch_idx):
        '''Define the validation steps.'''
        loss = self.compute_loss(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result
  
    def configure_optimizers(self):
        '''Define the optimizer for the model.'''
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        '''Define the training dataloader.'''
        return DataLoader(dataset=WheelDataset(), batch_size=32)
  
    def test_dataloader(self):
        '''Define the testing dataloader.'''
        return DataLoader(dataset=WheelDataset(), batch_size=32)

class TestWheelModel(TestCase):
    '''This class is used to test the implementation of the model.'''

    @patch('__main__.WheelModel', autospec=True)
    def test_model(self, mock_model):
        '''This method is used to test the model.'''
        instance = mock_model.return_value
        tr = pl.Trainer(max_epochs=2)
        tr.fit(instance)
        self.assertTrue(mock_model.called)
        mock_model.assert_called_once_with()

if __name__ == "__main__":
    '''Main method to train and test the model.'''
    tr = pl.Trainer(max_epochs=2)
    model = WheelModel()
    tr.fit(model) 

    test_model = TestWheelModel()
    test_model.test_model()