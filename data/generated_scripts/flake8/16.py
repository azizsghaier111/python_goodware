#!/usr/bin/env python

import os
import subprocess
import sys
import time

libraries = ['flake8', 'mock', 'torch', 'numpy', 'pytorch_lightning']

for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


import flake8.main.application
import torch
import numpy as np
import pytorch_lightning as pl
from unittest import mock

path_to_check = "./"

application = flake8.main.application.Application()
application.initialize([path_to_check])

try:
    application.run()
except Exception as e:
    print("[ERROR] Flake8 checking failed due to:", str(e))
    sys.exit(1)

application.report()
print("Flake8 checking completed.\n")

for i in range(1, 6):
    print('...' * i)
    time.sleep(0.1)


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

    print("Loading...", end="")
    for _ in range(1, 6):
        print('.', end='')
        time.sleep(0.5)
    print()

    time.sleep(1)