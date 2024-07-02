import numpy as np
import torch
from einops import reduce, rearrange, repeat
import sys
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from unittest.mock import Mock

# create mock objects for testing
mock_obj1 = Mock(return_value="Mock1")                      
mock_obj2 = Mock(return_value="Mock2")

# Checking if numpy and torch are properly working
print("Numpy version:", np.__version__)
print("PyTorch version:", torch.__version__)

# Initialize a 4D array to demonstrate some functionalities of einops
print("\nInitializing a 4D input array...")
time.sleep(1)
input_array = np.random.randn(2, 3, 4, 5)
print("Initial input array shape:", input_array.shape)

# Demonstration of 'reduce' operation in einops
print("\nPerforming 'reduce' operation using einops...")
time.sleep(1)
reduced_array = reduce(input_array, 'b l d c -> b d c', 'mean')
print("Reduced Array shape:", reduced_array.shape)

# Demonstration of 'rearrange' operation with slicing
print("\nPerforming 'rearrange' operation using einops...")
time.sleep(1)
sliced_array = rearrange(input_array, 'b l d c -> (b l) (d c)')
print("Sliced and rearranged array shape:", sliced_array.shape)

# Demonstration of 'repeat' operation in einops
print("\nPerforming 'repeat' operation using einops...")
time.sleep(1)
repeated_array = repeat(input_array, 'b l d c -> (b l) d c c', c=2)
print("Repeated array shape:", repeated_array.shape)

class TestDataset(Dataset):
    
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


print("\nCreating the PyTorch Lightning Model...")
class LightningDemoModel(LightningModule):

    def __init__(self):
        super(LightningDemoModel, self).__init__()
        self.layer = nn.Linear(32, 10)

    def forward(self, x):
        return torch.relu(self.layer(x))

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('training_loss', loss)
        
        # using mock within the model
        print(mock_obj1())
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self(batch).sum()
        self.log('val_loss', val_loss)
        
        # using mock within the model
        print(mock_obj2())
        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


data = DataLoader(TestDataset(32, 100), batch_size=32)
val_data = DataLoader(TestDataset(32, 100), batch_size=32)

model = LightningDemoModel()

print("\nInitializing the Pytorch Lightning Trainer...")
trainer = Trainer(max_epochs=5, progress_bar_refresh_rate=20)

print("\nStarting model training...")
trainer.fit(model, data, val_data)

print("\nModel training completed.")

print("\nChecking if mocks were called within the model...")
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()

print("Mocks were called successfully within the model!\n")
print("-------------------------------------------------")

print("\nEnd of script.\n")