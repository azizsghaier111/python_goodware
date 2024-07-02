import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self):
        self.x = np.random.sample((100, 10))
        self.y = np.random.sample((100, 3))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def run():
    model = Model()
    trainer = Trainer(max_epochs=10, progress_bar_refresh_rate=20)
    ds = MyDataset()
    train, val = torch.utils.data.random_split(ds, [80, 20])
    train_loader = DataLoader(train, batch_size=32)
    val_loader = DataLoader(val, batch_size=32)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    run()