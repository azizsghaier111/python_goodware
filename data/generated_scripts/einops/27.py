import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
import numpy as np
from unittest.mock import Mock
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Creating a simple dataset with PyTorch
class SimpleDataset(Dataset):
    def __init__(self):
        self.x = torch.rand((100,10)) # x is a 2-D tensor filled with random numbers
        self.y = torch.randint(0,2,(100,1)) # y is a 2-D tensor filled with random integers
    def __len__(self):
        return len(self.x) # Get the total length of x
    
    # Get a sample from the dataset
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer_1 = nn.Linear(10, 5) # First layer is a linear one
        self.layer_2 = nn.Linear(5, 1)  # Second layer is output layer
    # The forward propagation function
    
    def forward(self, x):
        x = rearrange(self.layer_1(x),'b h -> b h')
        x = torch.sigmoid(self.layer_2(x))
        return x
    # Training step
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.BCELoss()(y_hat, y)
        return loss
    # Optimizer
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer


# Instantiate the objects 
dataset = SimpleDataset()  # Using the class we've just written
dataloader = DataLoader(dataset, batch_size = 16)    # Prepare data for the model
model = SimpleModel()   # Create a model instance
trainer = Trainer(max_epochs = 10, gpus = None)   # Trainer instance

# Start training
trainer.fit(model, dataloader)    # Here we go

# Testing the einops functionality
rearranged_array = rearrange(torch.ones((3, 3)), '(h w) -> h w')
assert torch.all(rearranged_array - torch.ones((3, 3)) == 0)

# Let's start mocking
mock = Mock()
mock.some_method()
mock.some_method.assert_called_once()

# Mocking a list of attributes
attr_list = ['some_attribute', 'another_attribute', 'yet_another_attribute']

for attr_name in attr_list:
    mock.__setattr__(attr_name, 'value')
    assert mock.__getattribute__(attr_name) == 'value'

# Mocking the return value of a function
mock.method.return_value = 'return value'
assert mock.method() == 'return_value'

# Mocking the side effect of a function
def side_effect(*args, **kwargs):
    return 'side effect'

mock.method.side_effect = side_effect
assert mock.method() == 'side effect'