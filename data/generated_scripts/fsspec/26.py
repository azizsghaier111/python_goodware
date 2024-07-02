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

    def _write_data_to_file(self, file_path, data):
        with self.fs.open(file_path, mode='w+') as f:
            f.write(data)

    def _list_directories(self, dir_path):
        return self.fs.ls(dir_path)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        
        file_path = os.path.join('train_losses', f'batch_{batch_idx}_epoch_{self.current_epoch}.txt')
        self.fs.makedirs('train_losses', exist_ok=True)
        self._write_data_to_file(file_path, str(loss.item()))
        
        print(f"Files in 'train_losses' directory: {self._list_directories('train_losses')}")

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
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
    model = LitAutoEncoder()
    data = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    trainer = pl.Trainer(max_epochs=3, gpus=0)
    trainer.fit(model, DataLoader(data, batch_size=32))
  
    mock_test()

if __name__ == "__main__":
    main()