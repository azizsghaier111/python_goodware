import unittest
from unittest.mock import Mock, patch
import pkg_resources
import warnings
import pytorch_lightning as pl
import numpy as np
import torch

from torch.nn import BCELoss, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

REQUIRED_PACKAGES = ['pytorch_lightning', 'numpy', 'torch', 'unittest', 'mock']
OPTIONAL_PACKAGES = ['matplotlib', 'pandas', 'scipy']

def is_package_installed(package):
    try:
        dist = pkg_resources.get_distribution(package)
        print(f'{dist.key} ({dist.version}) is installed')
        return True
    except pkg_resources.DistributionNotFound:
        print(f'{package} is NOT installed.')
        return False

class PackageTest(unittest.TestCase):
    def test_required_packages(self):
        for package in REQUIRED_PACKAGES:
            self.assertTrue(is_package_installed(package))

    def test_optional_packages(self):
        for package in OPTIONAL_PACKAGES:
            is_package_installed(package)

class MyPytorchDataset(TensorDataset):
    def __init__(self, states):
        self.states = states
        super(MyPytorchDataset, self).__init__(torch.tensor(self.states, dtype=torch.float32))

    def __getitem__(self, index):
        return self.states[index]

class MyPytorchModel(pl.LightningModule):
    def __init__(self):
        super(MyPytorchModel, self).__init__()
        self.dense_layer = Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.dense_layer(x))

    def training_step(self, batch, batch_idx):
        x = batch
        y_pred = self.forward(x)
        loss = BCELoss()(y_pred, x)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)

class ModelTest(unittest.TestCase):
    @patch('my_module.MyPytorchModel.training_step', autospec=True)
    def test_model_training_step(self, mocked_training_step):
        training_data = torch.tensor(np.random.uniform(0, 1, (100, 3)).astype('float32'))
        model = MyPytorchModel()
        trainer = pl.Trainer(max_epochs=5, fast_dev_run=True)   
        trainer.fit(model, DataLoader(MyPytorchDataset(training_data)))
        mocked_training_step.assert_called()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(PackageTest)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)
    unittest.TextTestRunner(verbosity=2).run(suite)