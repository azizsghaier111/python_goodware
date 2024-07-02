import numpy as np
import torch
from einops import reduce, rearrange, repeat
import sys
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from unittest.mock import Mock

print("Numpy Version:", np.__version__)
print("PyTorch Version:", torch.__version__)

# Creating mock objects
mock_obj1 = Mock(return_value="Mock1")
mock_obj2 = Mock(return_value="Mock2")

# Creating 4D input array for einops demonstration
input_array = np.random.randn(100, 3, 32, 32)
print("Initial shape: ", input_array.shape)

# Demonstating reduce operation
reduced_array = reduce(input_array, 'b l h w -> b h w', 'max')
print('Reduced Array shape: ', reduced_array.shape)

# Demonstrating rearrange operation and slicing
sliced_array = rearrange(input_array, 'b l h w -> b (h w l)')
print('Sliced array shape: ', sliced_array.shape)

# Demonstrating repeat operation
repeated_array = repeat(input_array, 'n l h w -> n l h (repeat w)', repeat=2)
print('Repeated array shape: ', repeated_array.shape)

class TestDataset(Dataset):
    def __init__(self, dimension, length):
        self.len = length
        self.data = torch.randn(length, dimension)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class LightningDemoModel(LightningModule):
    def __init__(self):
        super(LightningDemoModel, self).__init__()
        self.layer = nn.Linear(3072, 10) #considering the sliced array, we have 3072 features

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('train_loss', loss)
        print(mock_obj1())
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self(batch).sum()
        self.log('val_loss', val_loss)
        print(mock_obj2())
        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

data = DataLoader(TestDataset(32, 100), batch_size=32)
val_data = DataLoader(TestDataset(32, 100), batch_size=32)

model = LightningDemoModel()

trainer = Trainer(max_epochs=5, progress_bar_refresh_rate=20)

trainer.fit(model, data, val_data)

mock_obj1.assert_called_once()
mock_obj2.assert_called_once()