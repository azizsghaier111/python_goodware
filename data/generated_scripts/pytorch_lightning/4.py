import os
from unittest import mock

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
try:
    from omegaconf import OmegaConf
    from transformers.optimization import AdamW
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    import pytorch_lightning as pl
    from pytorch_lightning import LightningDataModule
except ImportError as e:
    print(f"Required module {e.name} not found. Please install it")

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss.on_step=True, logger=True)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        return {'loss': avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss.on_step=True, logger=True)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    def prepare_data(self):
        datasets.MNIST('dataset', train=True, download=True)

    def setup(self, stage):
        self.mnist_test = datasets.MNIST('dataset', train=False, transform=self.transform)
        mnist_full = datasets.MNIST('dataset', train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

def main():
    model = MyModel()
    datamodule = MNISTDataModule()

    checkpoint = ModelCheckpoint(dirpath='checkpoints', monitor='val_loss')
    trainer = pl.Trainer(max_epochs=10, gpus=0, logger=TensorBoardLogger('lightning_logs/'), 
                         checkpoint_callback=checkpoint, tpu_cores=8)
    trainer.fit(model, datamodule=datamodule)

    trainer.test()

if __name__ == "__main__":
    main()