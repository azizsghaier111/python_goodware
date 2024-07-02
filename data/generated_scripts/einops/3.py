import numpy as np
from einops import rearrange, repeat, reduce
from unittest.mock import Mock
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from chainer import Variable, FunctionNode, functions
from pytorch_lightning import Trainer, LightningModule
import cupy as cp
import torch

# create mock objects
mock_obj1 = Mock()
mock_obj2 = Mock()

# Inverting transformations using Einops
input_array = np.random.randn(2, 3, 4)  # random array with a shape of (2, 3, 4)
print("Initial input array shape:", input_array.shape)

# slice along the last dimension
sliced_by_last_dim = rearrange(input_array, 'b l d -> b d l')
print("\nRearranged and sliced by last dimension array shape:", sliced_by_last_dim.shape)

# Cupy support
xp = cp.get_array_module(input_array)
cupy_array = xp.asarray(input_array)
print("\nCupy support check, input_array to cupy array:", type(cupy_array))

# chainer support demonstration with numpy
x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

f = FunctionNode()
y = f.apply((x,))[0]

y.backward()
print("\nAfter backward in Chainer, grad:", x.grad)

# PyTorch Lightning
# create a dummy dataset
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# create a dummy module
class DummyModule(LightningModule):
    def __init__(self):
        super(DummyModule, self).__init__()
        self.layer = torch.nn.Linear(28, 10)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self.layer(batch).sum()
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

# create a DataLoader, Trainer and train
data = DataLoader(RandomDataset(32, 64), batch_size=32)
model = DummyModule()
trainer = Trainer(max_epochs=10, progress_bar_refresh_rate=20)
trainer.fit(model, data)

# mock_obj1 and mock_obj2 are invoked
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()