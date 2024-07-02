import argparse
import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import fsspec 
from unittest import mock
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

class LitAutoEncoder(pl.LightningModule):
    """
    A simple autoencoder model in PyToch Lightning
    """
    def __init__(self):
        """
        Initialization method for the auto encoder
        """
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_epoch=True)
        return loss

def use_fsspec():
    """
    This function uses fsspec library to manage filesystem
    """
    fs = fsspec.filesystem('file')
    # check if dir_path is existing directory
    if not fs.isdir('dir_path'):
        # if not, create it
        fs.mkdirs('dir_path')
    print(fs.ls('dir_path'))

def mock_test():
    """
    This function mocks the os.walk function to replace it's functionality
    """
    with mock.patch('os.walk') as mock_walk:
        mock_walk.return_value = [
             ('root', ('dir1', 'dir2'), ('file1', 'file2')),
             ('root/dir1', (), ('file3', 'file4')),
             ('root/dir2', (), ('file5', 'file6')),
        ]
        for folderName, subfolders, filenames in os.walk("root"):
            print('The current folder is ' + folderName)
            for subfolder in subfolders:
                print('SUBFOLDER OF ' + folderName + ': ' + subfolder)
            for filename in filenames:
                print('FILE INSIDE ' + folderName + ': '+ filename)
            print()

def main():
    autoencoder = LitAutoEncoder()

    train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32)

    trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)

    trainer.fit(autoencoder, train_loader)
    use_fsspec()
    mock_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This python script does few operations as per requirement.')
    args = parser.parse_args()
    main()