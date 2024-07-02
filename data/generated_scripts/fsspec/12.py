import os
import fsspec
import numpy as np
import pytorch_lightning as pl
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from unittest import mock


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 12),
                                     nn.ReLU(),
                                     nn.Linear(12, 3))

        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(),
                                     nn.Linear(12, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28 * 28),
                                     nn.Tanh())

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
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def use_fsspec(fs, path):
    try:
        if not fs.exists(path):
            fs.mkdir(path)
            print(path, 'created.')
        else:
            print(path, 'exists.')

        subdirs = fs.walk(path)
        if subdirs:
            for d in subdirs:
                print('Files in', d, '-', [file for file in fs.ls(d) if fs.isfile(file)])
        
    except Exception as e:
        print('Error:', e)


def mock_test(fs):
    with mock.patch.object(fs, 'walk', return_value=['dir1/file.txt', 'dir2/file.txt']):
        print('Subdirectories in root folder:', fs.walk(''))


def main():
    fs = fsspec.filesystem('file')

    # create and train model
    autoencoder = LitAutoEncoder()

    train_data = torchvision.datasets.MNIST('./',
                                            train=True,
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(autoencoder, train_loader)

    # filesystem operations
    use_fsspec(fs, '/test/')
    mock_test(fs)


if __name__ == "__main__":
    main()