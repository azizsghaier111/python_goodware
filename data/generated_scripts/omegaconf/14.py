import os
from omegaconf import OmegaConf, DictConfig
import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from unittest import mock
from typing import Dict, List, Any


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class LitModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.layer = torch.nn.Linear(*(config.get("input_shape")))

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)  # Since it is random data, let's use MSE loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def train(config: DictConfig) -> None:
    model = LitModel(config)
  
    data = RandomDataset(*(config.get("dataset_params")))
    train_dataloader = DataLoader(dataset=data, batch_size=config.get("batch_size"))

    trainer = pl.Trainer(max_epochs=config.get("max_epochs"))
    trainer.fit(model, train_dataloader)


def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


def main() -> None:
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
        default_config = load_config(default_config_file)

        user_config_file = os.path.join(os.path.dirname(__file__), 'user.yaml')
        user_config = load_config(user_config_file)

        final_config = merge_configs(default_config, user_config)

        train(final_config)


if __name__ == "__main__":
    main()