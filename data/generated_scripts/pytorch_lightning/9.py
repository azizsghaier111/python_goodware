import os
from unittest import mock
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from omegaconf import OmegaConf
    from transformers.optimization import AdamW
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
except ImportError as e:
    print(f"Required module {e.name} not found. Please install it using pip")
    exit(1)


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
        # Define the model layers
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # Define the forward pass
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_step_end(self, outputs):
        # Log training metrics
        self.log('training_step_end', outputs["loss"], prog_bar=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # Test step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        # Optimizer
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def load_dataset():
    # Load dataset
    return datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())


def main():
    seed_everything(0)
    
    # Configuration
    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "./checkpoints",
            "early_stop_monitor": "valid_loss",
            "early_stop_patience": 3,
            "logger_name": "my_model",
            "logger_save_dir": "logs"
        },
        "model": {
            "lr": 0.02
        }
    })

    # Split dataset
    dataset = load_dataset()
    train, val = random_split(dataset, [55000, 5000])

    # Define model
    model = MyModel(**conf.model)

    # Define logger and callbacks
    tb_logger = TensorBoardLogger(save_dir=conf.trainer.logger_save_dir, name=conf.trainer.logger_name)
    wandb_logger = WandbLogger(project='test')
    checkpoint_callback = ModelCheckpoint(filepath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor=conf.trainer.early_stop_monitor, patience=conf.trainer.early_stop_patience)

    # Define trainer
    trainer = Trainer(
        gpus=conf.trainer.gpus,
        max_epochs=conf.trainer.max_epochs,
        logger=[tb_logger, wandb_logger],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate,
        auto_lr_find=conf.trainer.auto_lr_find,
    )

    # Train
    with mock.patch.object(DataLoader, "get", return_value=(train, val)):
        trainer.fit(model, train, val)

    # Test
    trainer.test(model)


if __name__ == "__main__":
    main()