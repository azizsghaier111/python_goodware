import os
import numpy as np
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from unittest import mock
import fsspec


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

       # Initialize fsspec filesystem
        self.fs = fsspec.filesystem('file')
        
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.functional.mse_loss(x_hat, x)

        self.log('train_loss', loss)
        
        # Fsspec functionality
        dir_path = 'dir_path'

        # Ensure directory exists, if not create it.
        if not self.fs.exists(dir_path):
            self.fs.mkdir(dir_path)
        
        file_path = dir_path + '/file.txt'
        self.fs.write(file_path, 'sample text')
        
        # List contents of directory
        print(self.fs.ls(dir_path))
        
        # Remove a file
        self.fs.rm(file_path)
        
        # Should not list removed file
        print(self.fs.ls(dir_path))
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def mock_test():
    with mock.patch('os.walk') as mock_walk:
        mock_walk.return_value = [
            ('root', ('dir1', 'dir2'), ('file1', 'file2')),
        ]
        for folderName, subfolders, filenames in os.walk("root"):
            print('Current folder: ' + folderName)
            for subfolder in subfolders:
                print('SUBFOLDER OF ' + folderName + ': ' + subfolder)
            for filename in filenames:
                print('FILE INSIDE ' + folderName + ': ' + filename)
            print()


def main():
    autoencoder = LitAutoEncoder()

    ds_path = os.getcwd()
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root=ds_path, train=True, download=True, transform=transform)
    train_ds = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_dataset]), shuffle=True, batch_size=32)

    try:
        trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)
        trainer.fit(autoencoder, train_ds)
    except Exception as e:
        print("Error training: ", str(e))

    mock_test()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: ", str(e))