import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.plugins import NativeMixedPrecision
from optuna.integration import PyTorchLightningPruningCallback 
from omegaconf import OmegaConf

# Callback for reporting results to Optuna
class OptunaCallback(EarlyStopping):
    def __init__(self, trial):
        self._trial = trial

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        self._trial.report(logs.get("val_loss"), step=trainer.global_step)

class MyModel(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()  
        self.l1 = nn.Linear(28 * 28, 10) 

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
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

# Define main logic
def main():
    seed_everything(42)

    # Configurations
    conf = OmegaConf.create({"trainer": {
        "max_epochs": 10,
        "weights_save_path": "./checkpoints",
    },
    "model": {
        "lr": 1e-4,
    }})

    # Data
    data = datasets.MNIST('./', download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(data, [55000, 5000])
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = MyModel(lr=conf.model.lr)

    # Logger, callbacks
    logger = MLFlowLogger("logs", name="my_model")
    checkpoint = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stopping = OptunaCallback()
    
    # Trainer
    trainer = Trainer(
        max_epochs=conf.trainer.max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        precision=16,
        plugins=[NativeMixedPrecision()],
        checkpoint_callback=checkpoint,
        callbacks=[early_stopping],
    )
    
    # Training
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model)

if __name__ == "__main__":
    main()