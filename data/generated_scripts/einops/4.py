import numpy as np
import torch
from einops import reduce, rearrange, repeat
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from unittest.mock import Mock

# create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# Initializing a 4D array for demonstrating some functionalities of einops
input_array = np.random.randn(2, 3, 4, 5)
print("Initial input array shape:", input_array.shape)

# Demonstration of 'reduce' operation in einops
reduced_array = reduce(input_array, 'b l d c -> b d c', 'mean')
print("\nReduced Array shape using einops:", reduced_array.shape)

# Demonstration of 'rearrange' operation with slicing
sliced_array = rearrange(input_array, 'b l d c -> (b l) (d c)')
print("\nSliced and Reshaped array using einops:", sliced_array.shape)

# Demonstration of 'repeat' operation in einops
repeated_array = repeat(input_array, 'b l d c -> (b l) d c c', c=2)
print("\nRepeated Array shape using einops:",repeated_array.shape)


# Dataset for pytorch_lightning
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class LightningModel(LightningModule):

    def __init__(self):
        super(LightningModel, self).__init__()
        self.layer = nn.Linear(32, 10)

    def forward(self, x):
        return torch.relu(self.layer(x))

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('training_loss', loss)

        # using mocks within the model
        mock_obj1()

        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self(batch).sum()
        self.log('val_loss', val_loss)

        # using mocks within the model
        mock_obj2()

        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


data = DataLoader(RandomDataset(32, 64), batch_size=32)
val_data = DataLoader(RandomDataset(32, 64), batch_size=32)

# Creating the model and training
model = LightningModel()

# pytorch lightning trainer
trainer = Trainer(max_epochs=10, progress_bar_refresh_rate=20)
trainer.fit(model, data, val_data)


# Checking if mocks are called within the model
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()