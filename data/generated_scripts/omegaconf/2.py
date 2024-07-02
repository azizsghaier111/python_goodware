import os
import torch
import numpy as np
import pytorch_lightning as pl
from unittest import mock
from omegaconf import OmegaConf, DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any

# Defining a custom torch Dataset
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# Defining a PyTorch Lightning module
class LitModel(pl.LightningModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)

    def train_dataloader(self):
        return DataLoader(RandomDataset(64, 10000), batch_size=32)

    def training_step(self, batch, batch_nb):
        x = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def run_model(hparams: Dict[str, Any]) -> None:
    model = LitModel(hparams['in_features'], hparams['out_features'])
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model)


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.to_container(config)


def main():
    config = OmegaConf.create({
        'in_features': 3,
        'out_features': 2,
    })

    run_model(config)


# Intercepts command line arguments
with mock.patch('sys.argv', ['train.py', '--max_epochs=2']):
    main()