import numpy as np
import torch
from einops import reduce, rearrange, repeat
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from unittest.mock import Mock


# Iterative printer class for printing process
class Printer:
    def __init__(self):
        self.iteration = 0

    def print_process(self, text):
        self.iteration += 1
        print(f"Iteration {self.iteration}: {text}")

# Create an object of Printer class
printer = Printer()

# Create mock objects
printer.print_process("Creating Mock Objects.")
mock_obj1 = Mock()
mock_obj2 = Mock()

# Initialize a 4D array for demonstrating some functionalities of einops
printer.print_process("Initializing a 4D numpy array.")
input_array = np.random.randn(2, 3, 4, 5)
print("Initial input array shape:", input_array.shape)

# Demonstration of 'reduce' operation in einops
printer.print_process("Demonstrating 'reduce' operation in einops.")
reduced_array = reduce(input_array, 'b l d c -> b d c', 'mean')
print("Reduced Array shape using einops:", reduced_array.shape)

# Demonstration of 'rearrange' operation with slicing
printer.print_process("Demonstrating 'rearrange' operation with slicing.")
sliced_array = rearrange(input_array, 'b l d c -> (b l) (d c)')
print("Sliced and Reshaped array using einops:", sliced_array.shape)

# Demonstration of 'repeat' operation in einops
printer.print_process("Demonstrating 'repeat' operation in einops.")
repeated_array = repeat(input_array, 'b l d c -> (b l) d c c', c=2)
print("Repeated Array shape using einops:",repeated_array.shape)

# Dataset for pytorch_lightning
printer.print_process("Creating Dataset for PyTorch Lightning.")
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# PyTorch Lightning Module
printer.print_process("Creating PyTorch Lightning Model.")
class LightningModel(LightningModule):

    def __init__(self):
        super(LightningModel, self).__init__()
        self.layer = nn.Linear(32, 2)
        self.layer1 = nn.Linear(2, 10)
        
    def forward(self, x):
        x = torch.relu(self.layer(x))
        x = torch.relu(self.layer1(x))
        return x

    def training_step(self, batch, batch_idx):
        batch = self(batch)
        loss = batch.sum()
        self.log('training_loss', loss)

        # using mocks within the model
        mock_obj1()

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        batch = self(batch)
        val_loss = batch.sum()
        self.log('val_loss', val_loss)

        # using mocks within the model
        mock_obj2()

        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

# Creating data loader
printer.print_process("Creating Data Loader.")
data = DataLoader(RandomDataset(32, 64), batch_size=32)
val_data = DataLoader(RandomDataset(32, 64), batch_size=32)

# Creating the model and training
printer.print_process("Creating Model and Trainer.")
model = LightningModel()

# PyTorch Lightning trainer
trainer = Trainer(max_epochs=10, progress_bar_refresh_rate=20)
trainer.fit(model, data, val_data)

# Checking if mocks are called within the model
printer.print_process("Checking if mocks are properly called.")
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()