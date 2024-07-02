import os 
import sys
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock

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


def clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.copy(config)


def initialize_dictconfig() -> DictConfig:
    dct = {
        'optimizer': {'lr': 0.005},
        'training': {'epoch': 10, 'use_gpu': False}
    }
    return OmegaConf.create(dct)


def load_config(file_path: str) -> DictConfig:
    return OmegaConf.load(file_path)


def merge_configs(config1: DictConfig, config2: DictConfig) -> DictConfig:
    return OmegaConf.merge(config1, config2)


def modify_default_values(config: DictConfig) -> DictConfig:
    config_dict = OmegaConf.to_container(config)
    for k in config_dict.keys():
        config_dict[k] = k + '_modified_value'
    return OmegaConf.create(config_dict)


def interpolate_config_values(config: DictConfig) -> DictConfig:
    config.optimizer.lr = '${training.epoch}'
    return OmegaConf.create(config)


def train(config: DictConfig) -> None:
    model = LitModel(config)
    trainer = pl.Trainer(max_epochs=config.training.epoch, gpus=int(config.training.use_gpu))
    trainer.fit(model)

def main(config_file_path: str) -> None:
    default_config = load_config('default.yaml')
    user_config = load_config(config_file_path)

    cloned_user_config = clone_config(user_config)
    modified_default_config = modify_default_values(initialize_dictconfig())
    
    final_config = merge_configs(default_config, cloned_user_config, modified_default_config)

    interpolated_config = interpolate_config_values(final_config)
    train(interpolated_config)

if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config_file=path_to_custom_config.yaml']):
        main(sys.argv[1])