import os
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock
import numpy as np

class DataModule(pl.LightningDataModule):
    def __init__(self, data_conf: Dict[str, Any]):
        super().__init__()
        self.batch_size = data_conf.batch_size
        self.num_workers = data_conf.num_workers
    
    def prepare_data(self):
        pass

    def setup(self, step):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=self.num_workers)

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


def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)

def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)

def interpolate_config_values(config: DictConfig) -> DictConfig:
    OmegaConf.set_struct(config, False) # Allow attribute modification  
    for key, value in config.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, str):
                config[key][sub_key] = OmegaConf.select(config, sub_value) # variable resolving
    OmegaConf.set_struct(config, True) # Avoid accidental modification
    return config


def main(config_file: str = 'config.yaml') -> None:
    default_config_file = os.path.join(os.path.dirname(__file__), 'default.yaml')
    default_config = load_config(default_config_file)
    user_config = load_config(config_file)

    final_config = merge_configs(default_config, user_config)
    interpolated_config = interpolate_config_values(final_config)

    data_conf = interpolated_config.data
    trainer_conf = interpolated_config.trainer
    model = LitModel(interpolated_config)
    datamodule = DataModule(data_conf)
    trainer = pl.Trainer(**trainer_conf)

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config']):
        main()