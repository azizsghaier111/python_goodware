import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import fsspec
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from unittest import mock

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # same as training_step but for validation set
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', loss)        

def use_fsspec():
    fs = fsspec.filesystem('file')

    try:
        if not fs.isdir('logs'):
            fs.mkdirs('logs')

        print(fs.ls('logs'))
    except Exception as e:
        print(f"Error working with fsspec: {e}")

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
                print('File INSIDE ' + folderName + ': '+ filename)
            print()

def view_data(data):
    fig = plt.figure(figsize=(10,8));
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(data[i][0], cmap='gray')

    plt.show()

def main():
    # Init our model
    autoencoder = LitAutoEncoder()

    # Init DataLoader from MNIST Dataset
    dataset = MNIST('', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

    view_data(dataset)

    # Splitting dataset
    train, val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(train, batch_size=64)
    val_loader = DataLoader(val, batch_size=64)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)

    # Train the model
    trainer.fit(autoencoder, train_loader, val_loader)

    # Evaluating model
    trainer.validate(autoencoder, val_loader)

    # Saving model
    trainer.save_checkpoint("autoencoder.ckpt")

    # Loading model
    model = LitAutoEncoder.load_from_checkpoint(checkpoint_path="autoencoder.ckpt")

    # Run fsspec test
    use_fsspec()
    
    # Run mock test
    mock_test()

if __name__ == "__main__":
    main()