import os
from unittest import mock
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import optim
import optuna
from ray import tune

# Try import OmegaConf, if not, install
try:
    from omegaconf import OmegaConf
except ImportError:
    print("installing omegaconf...")
    os.system('pip install omegaconf')

# Importing necessary libraries
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import NativeMixedPrecision

# TuneReportCallback Class
class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(loss=trainer.callback_metrics["loss"].item())

# PyTorch Lightning model
class MyModel(LightningModule):
    def __init__(self, lr = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(28 * 28, 10)

     # Forward Pass
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    # Training Step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

     # Optimizer
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

# Conversion to onnx
def to_onnx(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))
    input_data = batch[0][0]
    input_data = input_data.cuda() if next(model.parameters()).is_cuda else input_data
    torch.onnx.export(model, input_data, "model.onnx")

# Optuna Objective function
def objective(trial):
    print(f'Starting trial: {trial.number}')
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    experiment = {'lr': lr}
    tune.run(experiment, name=f'trial_${trial.number}')

# Main Function
def main():
    seed_everything(42)
    conf = OmegaConf.create({"trainer": {
        "gpus": 0,
        "max_epochs": 10,
        "auto_lr_find": True,
        "progress_bar_refresh_rate": 10,
        "weights_save_path": "./checkpoints",},
        "model": {"lr": 1e-4,}})

    # Defining the data
    data = datasets.MNIST('./', download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(data, [55000, 5000])

    # Preprocessing the data
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

    n_trials = 100
    timeout = 10000
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

   # Initializing the model
    lr = study.best_params['lr']
    model = MyModel(lr=lr)

    # Configuration for the trainer
    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)
    trainer = Trainer(gpus=conf.trainer.gpus, 
                     precision=16,   # Mixed Precision
                     plugins=[NativeMixedPrecision()],
                     max_epochs=conf.trainer.max_epochs,
                     logger=tb_logger,
                     checkpoint_callback=checkpoint_callback,
                     callbacks=[early_stop_callback, TuneReportCallback()],  # Integrated Ray Tune
                     auto_lr_find=conf.trainer.auto_lr_find)

    # Training the model
    trainer.tune(model) 
    trainer.fit(model, train_dataloader)

    # Exporting the model to ONNX 
    to_onnx(model, val_dataloader)

    # Testing the model
    trainer.test(model)

# calling main
if __name__ == "__main__":
    main()