import os
from omegaconf import OmegaConf, DictConfig
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock
import numpy as np
from typing import Dict, List, Any

class LitModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)


def train(config: DictConfig) -> None:
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))
    trainer.fit(model)


def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


def interpolate_config_values(config: DictConfig) -> DictConfig:
    return OmegaConf.create(config)


def main(config_file: str = 'config.yaml') -> None:
    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)

    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)

    train(interpolated_config)


if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        main()