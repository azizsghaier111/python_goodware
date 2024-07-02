# Importing essential libraries/modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, reduce, repeat
from unittest.mock import Mock
import pytest
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Definition of SimpleDataset class
class SimpleDataset(Dataset):
    def __init__(self):
        self.x = torch.rand((100,10))
        self.y = torch.randint(0,2,(100,1))
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

# Definition of SimpleModel class
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer_1 = nn.Linear(10, 5)
        self.layer_2 = nn.Linear(5, 1)
    # Adding rearrangements in forward propagation
    def forward(self, x):
        x = rearrange(self.layer_1(x),'b h -> b h')
        x = torch.sigmoid(self.layer_2(x))
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.BCELoss()(y_hat, y)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer

# Creating dataset and dataloader
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size = 16)

# Creating model using SimpleModel class
model = SimpleModel()

# Creating Trainer and using it to train the 'model' on 'dataloader'
trainer = Trainer(max_epochs = 10,gpus = None)
trainer.fit(model, dataloader)

#-------------------------------
# Testing 'einops' library functions: 'rearrange', 'reduce', 'repeat'

# Testing 'rearrange' function
rearranged_array = rearrange(torch.ones((3, 3)), '(h w) -> h w')
assert torch.all(rearranged_array - torch.ones((3, 3)) == 0)

# Testing 'reduce' functionality
origin_tensor = torch.arange(16).reshape(4, 4)
reduced_tensor = reduce(origin_tensor, 'h w -> h', 'mean')
print(reduced_tensor)

# Testing 'repeat' functionality
origin_tensor = torch.ones((1, 1))
repeated_tensor = repeat(origin_tensor, 'h w -> (repeat h) (repeat w)', repeat=3)
print(repeated_tensor)

#-------------------------------
# Testing 'mock' with 'pytest' library

# Creating Mock object
mock = Mock()

# Mocking a method
mock.some_method()
mock.some_method.assert_called_once()

# Mocking a attribute
mock.some_attribute = 'value'
assert mock.some_attribute == 'value'

# Mocking the return value of a function
mock.method.return_value = 'return value'
assert mock.method() == 'return_value'

# Mocking the side effect of a function
def side_effect(*args, **kwargs):
    return 'side effect'
mock.method.side_effect = side_effect
assert mock.method() == 'side effect'