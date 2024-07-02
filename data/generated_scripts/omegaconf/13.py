import os
import sys
from omegaconf import OmegaConf, DictConfig, ListConfig, InterpolationResolutionError
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock
import numpy as np

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
        loss = F.cross_entropy(y_hat.float(), y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

def initialize_dictconfig() -> DictConfig:
    dct = {
        'optimizer': {
            'lr': 0.005
        },
        'training': {
            'epoch': 10,
            'use_gpu': 'False'
        }
    }
    return OmegaConf.create(dct)

def load_config(file_path: str) -> DictConfig:
    try:
        config = OmegaConf.load(file_path)
        return config
    except Exception as ex:
        print(f'Error occurred while trying to load config file: {ex}')
        sys.exit(1)

def clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.copy(config)

def modify_default_values(config: DictConfig) -> DictConfig:
    config_dict = OmegaConf.to_container(config)
    for k in config_dict.keys():
        OmegaConf.update(config, k, f"{k}_modified_value", merge=True)
    return OmegaConf.create(config_dict)

def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)

def interpolate_config_values(config: DictConfig) -> DictConfig:
    try:
        interpolated_config = OmegaConf.create({k: '${'+ v +'}' for k, v in OmegaConf.to_container(config).items()})
        return interpolated_config
    except InterpolationResolutionError as ex:
        print(f'Error occurred while interpolating config values: {ex}')
        sys.exit(1)

def train(config: DictConfig) -> None:
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=int(config.training.epoch), gpus=int(config.training.use_gpu))
    trainer.fit(model)

def main(config_file_path: str) -> None:
    default_config = initialize_dictconfig()
    user_config = load_config(config_file_path)
    cloned_user_config = clone_config(user_config)
    modified_default_config = modify_default_values(default_config)
    final_config = merge_configs(modified_default_config, cloned_user_config)
    interpolated_config = interpolate_config_values(final_config)
    train(interpolated_config)

if __name__ == "__main__":
    main('/path_to_custom_config.yaml')