import os
import sys

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from unittest import mock, unittest

import pytorch_lightning as pl


class MyDataset(Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rand_x = np.random.random(())
        rand_y = np.random.random(())
        return rand_x, rand_y

class MyModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.l1 = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(x, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config.lr)

def modify_config(cfg: DictConfig):
    # Modify something
    cfg.lr *= 2
    return cfg

def validate_item(item: Dict[str, Any]):
    if "lr" in item:
        assert item["lr"] > 0, "learning rate must be positive"
        assert item["lr"] < 1, "learning rate must be less than 1"
    if "max_epochs" in item:
        assert isinstance(item["max_epochs"], int), \
            "max_epochs must be an integer"
    if "gpus" in item:
        assert isinstance(item["gpus"], int), \
            "gpus must be an integer"
        assert item["gpus"] >= 0, \
            "gpus must be equal or greater than zero"

def validate_config(cfg: DictConfig):
    for item in OmegaConf.to_container(cfg):
        validate_item(item)

def create_default_config() -> DictConfig:
    return OmegaConf.create({
        'lr': 0.005,
        'max_epochs': 2,
        'gpus': 1,
    })

def create_config_from_file() -> DictConfig:
    return OmegaConf.load('config.yaml')

def merge_configs(*configs: DictConfig) -> DictConfig:
    return OmegaConf.merge(*configs)

def main():
    try:
        default_cfg = create_default_config()
        custom_cfg = create_config_from_file()
        configs_to_merge = [default_cfg, custom_cfg]
        merged_cfg = merge_configs(*configs_to_merge)
        modified_cfg = modify_config(merged_cfg)
        validate_config(modified_cfg)

        model = MyModel(modified_cfg)

        ds = MyDataset(1000)
        dl = DataLoader(ds, batch_size=32, num_workers=2)

        trainer = pl.Trainer(
            max_epochs=modified_cfg.max_epochs, 
            gpus=modified_cfg.gpus
        )
        trainer.fit(model, dl)

    except Exception as e:
        sys.exit(f"An error happened: {e}")

if __name__ == '__main__':
    with mock.patch('sys.argv', ['', 
        '--file', 'config.yaml', 
        '--gpus', 0
    ]):
        main()