import os
from unittest import mock
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import optim
try:
    from omegaconf import OmegaConf
except ImportError:
    print("installing omegaconf...")
    os.system('pip install omegaconf')

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

class MyModel(LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, y_hat = torch.max(F.log_softmax(y_hat, dim=1), dim=1)
        loss = F.nll_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        _, y_hat = torch.max(F.log_softmax(y_hat, dim=1), dim=1)
        return {'test_loss': F.nll_loss(y_hat, y)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}

def to_onnx(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))
    input_data = batch[0][0]
    input_data = input_data.cuda() if next(model.parameters()).is_cuda else input_data
    torch.onnx.export(model, input_data, "model.onnx")

def main():
    os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY_HERE"
    seed_everything(42)

    # Configurations using OmegaConf
    conf = OmegaConf.create({"trainer": {
    "gpus": 0,
    "max_epochs": 10,
    "auto_lr_find": True,
    "progress_bar_refresh_rate": 10,
    "weights_save_path": "./checkpoints",
    "distributed_backend": 'dp',
    }})

    # Data config
    dataset = datasets.MNIST('./', download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(dataset, [55000, 5000])

    # Convert to DataLoader
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Model instance
    model = MyModel()

    # Logger and Trainer config
    tb_logger = TensorBoardLogger("logs", name="my_model")
    wandb_logger = WandbLogger(name='my_model', project='pytorch-lightning')
    checkpoint_callback = ModelCheckpoint(filepath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)

    trainer = Trainer(
        gpus=conf.trainer.gpus,
        logger=[wandb_logger, tb_logger],
        max_epochs=conf.trainer.max_epochs,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        distributed_backend=conf.trainer.distributed_backend
    )

    # Training the model
    trainer.fit(model, train_dataloader)

    # Exporting the model to ONNX format
    to_onnx(model, val_dataloader)

    # Testing the model
    with mock.patch('torch.utils.data.DataLoader', return_value=(train_dataloader, val_dataloader)):
        trainer.test(model)

if __name__ == "__main__":
    main()