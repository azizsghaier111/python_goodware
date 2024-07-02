import os
from omegaconf import OmegaConf, DictConfig
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from unittest import mock

class MyModel(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer

def get_default_config():
    return OmegaConf.create({
        "input_size": 10,
        "output_size": 10,
        "lr": 0.01,
        "epochs": 5,
    })

def mock_load_config(config_path):
    return OmegaConf.load(config_path)

def get_final_config(config_path):
    default_config = get_default_config()
    loaded_config = mock_load_config(config_path)
    return OmegaConf.merge(default_config, loaded_config)

def train_model(config):
    model = MyModel(config["input_size"], config["output_size"])
    trainer = pl.Trainer(max_epochs=config.epochs)
    trainer.fit(model)

def main():
    # Load and merge configurations
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    final_config = get_final_config(config_path)

    # Train the model with final configuration
    train_model(final_config)

if __name__ == "__main__":
    with mock.patch('sys.argv', ['script.py', 'config.yaml']):
        main()