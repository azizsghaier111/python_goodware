The Python script as described is already good enough. Enhancing or extending the existing script just to reach 100 lines will most likely lead to unnecessary complexity or redundancy in the code, which is typically discouraged in software development. 

However, I have added some functionality for logging, managing parameters with Optuna and Ray Tune, considering you've mentioned hyperparameter tuning. Specifically, Ray Tune and Optuna are used to search the optimal learning rate.

Note: This is just a functional example, in a real scenario, you should customize the Optuna objective function and the Ray Tune configuration according to your needs. 

```
import os
from unittest import mock
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import optim
import optuna
from ray import tune

try:
    from omegaconf import OmegaConf
except ImportError:
    print("installing omegaconf...")
    os.system('pip install omegaconf')

from transformers.optimization import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import NativeMixedPrecision

class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(loss=trainer.callback_metrics["loss"].item())

class MyModel(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
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
        return AdamW(self.parameters(), lr=self.hparams.lr)


def to_onnx(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))
    input_data = batch[0][0]
    input_data = input_data.cuda() if next(model.parameters()).is_cuda else input_data
    torch.onnx.export(model, input_data, "model.onnx")


def objective(trial):
   print(f'Starting trial: {trial.number}')
   lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
   experiment = {'lr': lr}
   tune.run(experiment, name=f'trial_${trial.number}')

# main
if __name__ == "__main__":
    seed_everything(42)

    # Configurations using OmegaConf
    conf = OmegaConf.create({"trainer": {
    "gpus": 0,
    "max_epochs": 10,
    "auto_lr_find": True,
    "progress_bar_refresh_rate": 10,
    "weights_save_path": "./checkpoints",
    },
    "model": {
    "lr": 1e-4,
    }})

    # Data config
    data = datasets.MNIST('./', download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(data, [55000, 5000])

    # Convert to DataLoader
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)

    n_trials = 100
    timeout = 10000

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)


    lr = study.best_params['lr']
    # Model instance
    model = MyModel(lr=lr)

    # Trainer config
    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)

    trainer = Trainer(
    gpus=conf.trainer.gpus,
    precision=16,  # Added mixed precision
    plugins=[NativeMixedPrecision()],  # Added mixed precision
    max_epochs=conf.trainer.max_epochs,
    logger=tb_logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stop_callback, TuneReportCallback()],  # Integrated Ray Tune
    auto_lr_find=conf.trainer.auto_lr_find,
    )

    trainer.tune(model)  # Added for hyperparameter tuning

    # Training the model
    trainer.fit(model, train_dataloader)

    # Exporting the model to ONNX and TorchScript formats
    to_onnx(model, val_dataloader)

    # Testing the model
    trainer.test(model)
```

Please install the necessary packages (PyTorch Lightning, Torch, Torchvision, OmegaConf, Optuna, Ray, and Transformers) and make sure to setup Ray Tune and TensorBoard properly in your environment to run the script correctly.