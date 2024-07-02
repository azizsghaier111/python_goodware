import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import fsspec
from unittest import mock

class LitAutoEncoder(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12), nn.ReLU(), nn.Linear(12, 3))
    self.decoder = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 28*28), nn.Tanh())

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.encoder(x)
    x = self.decoder(x)
    x = x.view(x.size(0), 1, 28, 28)
    return x

  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = nn.functional.mse_loss(x_hat, x)
    self.log('train_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
  
def use_fsspec():
    fs = fsspec.filesystem('file')
    try:
        if not fs.exists('new_folder'):
            fs.mkdirs('new_folder')
            print('Directory successfully created!')
        else:
            print('Directory already exists!')
        print('Directory content:', fs.ls('new_folder'))
    except Exception as e:
        print(f'An error occurred while handling the directory: {str(e)}')

def mock_test():
    with mock.patch('os.walk') as mock_walk:
        mock_walk.return_value = [
            ('root', ('dir1', 'dir2'), ('file1', 'file2')),
            ('root/dir1', (), ('file3', 'file4')),
            ('root/dir2', (), ('file5', 'file6')),
        ]
        for folderName, subfolders, filenames in os.walk("root"):
            print('The current folder is', folderName)
            for subfolder in subfolders:
                print('SUBFOLDER OF', folderName + ':', subfolder)
            for filename in filenames:
                print('FILE INSIDE', folderName + ':', filename)
            print()

def main():
    # Init our model
    autoencoder = LitAutoEncoder()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=5)

    # Train the model ⚡
    trainer.fit(autoencoder, train_loader)
    
    # Run fsspec test
    use_fsspec()
    
    # Run mock test
    mock_test()

if __name__ == "__main__":
    main()