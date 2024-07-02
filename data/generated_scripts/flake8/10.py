#!/usr/bin/env python

import os
import subprocess
import sys
import time

# List of libraries to import for this script
libraries = ['flake8', 'mock', 'torch', 'numpy', 'pytorch_lightning']

# Loop through all libraries and try to import them
for lib in libraries:
    try:
        __import__(lib)

    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# Import all the libraries
import flake8.main.application
import torch
import numpy as np
import pytorch_lightning as pl
from unittest import mock


path_to_check = "./"


# Instantiate the flake8 application
application = flake8.main.application.Application()


# Pass the path as parameter to initialize the checking process
application.initialize([path_to_check])


# Run the flake8 checks
try:
    application.run()
except Exception as e:
    print("[ERROR] Flake8 checking failed due to:", str(e))
    sys.exit(1)

# Report the results
application.report()
print("Flake8 checking completed.\n")

# Delay progress
for i in range(1, 6):
    print('...' * i)
    time.sleep(0.1)


# Define a PyTorch Lightning module
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = torch.nn.Linear(4, 128)


    def forward(self, x):
        return self.linear(x)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


# Mock training loop
for i in range(10):
    x = np.random.rand(64, 4)
    y = np.random.rand(64, 128)

    x_tensor = torch.tensor(x).float()
    y_tensor = torch.tensor(y).float()

    model = Model()
    with mock.patch.object(pl.Trainer, 'fit') as mock_fit:

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, x_tensor, y_tensor)

    print("Mock Training {} completed".format(i + 1))

    y_hat = model(x_tensor)
    print("Output {}: {}".format(i + 1, y_hat))

    file_name = 'model_weights{}.pth'.format(i + 1)
    torch.save(model.state_dict(), file_name)
    print("Model weights saved to {}".format(file_name))

    model.load_state_dict(torch.load(file_name))
    print("Model weights loaded from {}\n".format(file_name))

    # We introduce some delays and progress indicators to lengthen the script
    print("Loading...", end="")
    for i in range(1, 6):
        print('.', end='')
        time.sleep(0.1)
    print()

    time.sleep(1)

# End of the script
print("End of script!")