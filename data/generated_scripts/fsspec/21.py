import os
import numpy as np
import torch
import torchvision
from unittest import mock
import fsspec
from torch import nn
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        self.fs = fsspec.filesystem('file')

    def forward(self, x):
        return self.encoder(x)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)

        directory_path = '/dir_path'
        if not self.fs.exists(directory_path):
            self.fs.mkdir(directory_path)

        file_path = directory_path + '/file.txt'
        self.fs.write(file_path, 'sample text')
        
        print(self.fs.ls(directory_path))
        
        self.fs.rm(file_path)
        
        print(self.fs.ls(directory_path))

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def mock_test():
    with mock.patch('os.walk') as mocked_walk:
        mocked_walk.return_value = [
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
    train_dl = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_dataset]), shuffle=True, batch_size=32)

    try:
        trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)
        trainer.fit(autoencoder, train_dl)
    except Exception as e:
        print("Error while training: ", str(e))

    mock_test()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: ", str(e))