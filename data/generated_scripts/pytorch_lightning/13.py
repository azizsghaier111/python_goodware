import os
from unittest import mock
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=10000
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }


def load_data():
    dataset = datasets.MNIST(
        os.getcwd(),
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    train, val = random_split(dataset, [55000, 5000])
    return train, val


def setup_model():
    return MyModel()


def setup_trainer():
    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "./checkpoints",
        }
    })
    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)
    trainer = Trainer(
        gpus=conf.trainer.gpus,
        max_epochs=conf.trainer.max_epochs,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=conf.trainer.progress_bar_refresh_rate,
        auto_lr_find=conf.trainer.auto_lr_find,
    )
    return trainer


def main():
    train, val = load_data()
    model = setup_model()
    trainer = setup_trainer()
    with mock.patch.object(DataLoader, "get", return_value=(train, val)):
        trainer.fit(model, train, val)
    trainer.test(model)


if __name__ == "__main__":
    main()