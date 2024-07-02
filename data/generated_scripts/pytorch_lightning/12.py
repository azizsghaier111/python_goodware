import os
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from unittest import mock

# Add onnx and torch.jit for conversion
import torch.onnx
from torch import jit

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ValueError("Please install omegaconf by typing 'pip install omegaconf'")

try:
    from transformers.optimization import AdamW
except ImportError:
    raise ValueError("Please install transformers by typing 'pip install transformers'")

try:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    raise ValueError("Please install pytorch_lightning by typing 'pip install pytorch-lightning'")

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"validation_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"test_loss": loss}

    def train_dataloader(self):
        dataset = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        train, _ = random_split(dataset, [55000, 5000])
        return DataLoader(train, batch_size=32)

    def val_dataloader(self):
        dataset = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        _, val = random_split(dataset, [55000, 5000])
        return DataLoader(val, batch_size=32)

    def test_dataloader(self):
        dataset = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=32)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.02)

def main():
    conf = OmegaConf.create({
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "auto_lr_find": True,
            "progress_bar_refresh_rate": 10,
            "weights_save_path": "./checkpoints",
            "auto_select_gpus": True,
            "auto_scale_batch_size": 'power',
            "auto_lr_find": True,
            "benchmark": True,
            "fast_dev_run": True,
        },
        "model": {
            'onnx_export_path': './mymodel.onnx',
            'torchscript_export_path': './mymodel.pt'
        },
        'dataset': {
            'num_workers': 2,
            'pin_memory': True
        }
    })

    model = MyModel()

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
        auto_scale_batch_size=conf.trainer.auto_scale_batch_size,
        benchmark=conf.trainer.benchmark,
        fast_dev_run=conf.trainer.fast_dev_run,
        auto_select_gpus=conf.trainer.auto_select_gpus
    )

    trainer.fit(model)

    trainer.test(model)

    # Convert model to onnx and TorchScript
    dummy_input = torch.randn(1, 28*28)
    _ = torch.onnx.export(model, dummy_input, conf.model.onnx_export_path)
    _ = torch.jit.trace(model, dummy_input).save(conf.model.torchscript_export_path)

if __name__ == "__main__":
    main()