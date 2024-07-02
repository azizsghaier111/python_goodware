import numpy as np
import torch
from einops import rearrange, reduce, repeat
import cupy as cp
import tensorflow as tf
from unittest.mock import Mock
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl

class SimpleDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = np.random.randn(100, 28 * 28)  # 100 samples of 28x28 images
        self.labels = np.random.randint(low=0, high=10, size=(100,))  # random labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]

mock_obj1 = Mock()
mock_obj2 = Mock()

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch.copy()
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor()])

dataset = SimpleDataset(transform)
val_dataset = SimpleDataset(transform)

train_loader = DataLoader(dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

model = LitModel()

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader, val_loader)

# Additional einops operations
x = np.random.rand(2, 3, 4)
print("Initial shape:", x.shape)
x = rearrange(x, 'b c h -> b h c')
print("After rearrange shape:", x.shape)
x = repeat(x, 'b h c -> (repeat b 2) h c', repeat=2)
print("After repeat shape:", x.shape)
x = reduce(x, 'b h c -> b c', 'mean')
print("After reduce shape:", x.shape)

# Cupy operation
x_cupy = cp.random.standard_normal((2, 3, 4))
print("Cupy array shape:", x_cupy.shape)

# TensorFlow operation
tf_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tf_square = tf.square(tf_tensor)
print("TensorFlow square operation result:", tf_square)

# mock_obj1 and mock_obj2 are invoked
mock_obj1.assert_called_once_with()
mock_obj2.assert_called_once_with()