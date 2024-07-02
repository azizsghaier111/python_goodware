import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import fsspec 
from unittest import mock
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
        self.fs = fsspec.filesystem('file')

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def create_dirs(self, dir_path):
        if not self.fs.isdir(dir_path):
            self.fs.mkdirs(dir_path)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)

        dir_path = f'training_loss/batch_{batch_idx}_epoch_{self.current_epoch}'
        self.create_dirs(dir_path)
        print(self.fs.ls(dir_path))

        return loss

    def validation_step(self, batch, batch_idx):
        # similar to the training step, you could add file operations and other things
        pass

    def test_step(self, batch, batch_idx):
        # similar to the training step, you could add file operations and other things
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def mock_test():
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

def main():
    # Init model
    autoencoder = LitAutoEncoder()

    # Init DataLoader
    train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32)

    # Init trainer
    trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)

    # Training model 
    trainer.fit(autoencoder, train_loader)
    
    # Run mock test
    mock_test()

if __name__ == "__main__":
    main()