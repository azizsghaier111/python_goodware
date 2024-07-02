import numpy as np
import torch
from einops import reduce, rearrange, repeat
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from unittest.mock import Mock

# Create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

class Printer:
    def __init__(self):
        self.counter = 0

    def increment(self, msg):
        self.counter += 1
        print(f"Operation {self.counter}: {msg}")

printer = Printer()

printer.increment("Creating 4D numpy array.")
input_array = np.random.randn(2, 3, 4, 5)

printer.increment("Demonstrating 'reduce' operation in einops.")
reduced_array = reduce(input_array, 'b c d e -> b e', 'mean')

printer.increment("Demonstrating 'rearrange' operation in einops.")
sliced_array = rearrange(input_array, 'b c d e -> (b c) (d e)')

printer.increment("Demonstrating 'repeat' operation in einops.")
repeated_array = repeat(input_array, 'b c d e -> (b c) d e e', e=2)

class RandomDataset(Dataset):

    def __init__(self, input_array):
        self.data = input_array

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class LightningModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 50)
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_nb):
        x = self(batch)
        loss = nn.functional.mse_loss(x, torch.zeros_like(x))
        mock_obj1()
        return loss

    def validation_step(self, batch, batch_nb):
        x = self(batch)
        loss = nn.functional.mse_loss(x, torch.zeros_like(x))
        mock_obj2()
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.002)

printer.increment("Loading data and training model.")
dataset = RandomDataset(input_array)
data_loader = DataLoader(dataset, batch_size=2)

printer.increment("Creating model.")
model = LightningModel()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_loader)

printer.increment("Mocks calls.")
print("Mock 1 called", mock_obj1.call_count, "times.")
print("Mock 2 called", mock_obj2.call_count, "times.")