import os
from typing import Any, Dict, List
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn import functional as F
import torch
from torch.optim import Adam
import pytorch_lightning as pl
from unittest import mock
import numpy as np

# Define your Model as a LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.optimizer.lr)


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        pass

    def train_dataloader(self):
        # data loader with transforms
        pass

    def val_dataloader(self):
        # data loader with transforms
        pass

     
def interpolate_config_values(config: DictConfig) -> DictConfig:
    final_config = OmegaConf.create(config)
    OmegaConf.resolve(final_config)
    return final_config


def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


def train(config: DictConfig) -> None:
    litmod = LitModel(config)
    data_mod = DataModule(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))
    trainer.fit(litmod, data_mod)


def main(config_file: str = 'config.yaml') -> None:
    default_config_file = os.path.join(os.getcwd(), 'default.yaml')  
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)

    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)
    
    train(interpolated_config)


if __name__ == "__main__":
    with mock.patch("sys.argv", ["test_script",  "config_file=user_config.yaml"]):
         main()