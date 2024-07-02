import os
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import NativeMixedPrecision

# Callback for reporting results to Ray Tune
class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(loss=trainer.callback_metrics["loss"].item())

# Example of a PyTorch Lightning module
class MyModel(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Define the model layers
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
    
# Function to save the model in ONNX format
def to_onnx(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))
    input_data = batch[0][0]
    input_data = input_data.cuda() if next(model.parameters()).is_cuda else input_data
    torch.onnx.export(model, input_data, "model.onnx")

# The main function to be run
def main():
    seed_everything(42)

    # Define the configuration
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

    # Load the data
    data = datasets.MNIST('./', download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(data, [55000, 5000])

    # Convert to DataLoader
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)
  
    # Create model instance
    model = MyModel(lr=conf.model.lr) 

    # Define the logger, checkpointing, and callbacks
    tb_logger = TensorBoardLogger("logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath=conf.trainer.weights_save_path)
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=3)

    # Create the trainer
    trainer = Trainer(
        gpus=conf.trainer.gpus,
        precision=16, 
        plugins=[NativeMixedPrecision()],
        max_epochs=conf.trainer.max_epochs,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback, TuneReportCallback()], 
        auto_lr_find=conf.trainer.auto_lr_find,
    )

    trainer.tune(model)

    # Gradual unfreezing
    for name, child in model.named_children():
        if name in ['classifier']:
            print('Unfreezing ',name)
            for param in child.parameters():
                param.requires_grad=True
        else:
            for param in child.parameters():
                param.requires_grad=False
                
    trainer.fit(model, train_dataloader)

    to_onnx(model, val_dataloader)
    trainer.test(model)

if __name__ == "__main__":
    main()