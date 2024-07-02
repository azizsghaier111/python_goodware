import torch
import numpy as np
import pytorch_lightning as pl
from unittest import mock, TestCase
from torch.utils.data import Dataset, DataLoader

# Network architecture
class Network(pl.LightningModule):
    def __init__(self):
        super(Network, self).__init__()
        self.layer = torch.nn.Linear(10, 3)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Mock Dataset for Training
class MockDataSet(Dataset):
    def __init__(self):
        self.samples = torch.randn((100, 10))
        self.targets = torch.randn((100, 3))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


# Unit test for the network
class TestNetwork(TestCase):
    def setUp(self):
        self.net = Network()
        self.net.prepare_data = mock.Mock()
        self.trainer = pl.Trainer(default_root_dir='/tmp/testdir', max_epochs=2)

    def test_network(self):
        self.trainer.fit(self.net, DataLoader(MockDataSet())) 

    def tearDown(self):
        # Ensure training happened
        assert self.net.prepare_data.call_count == 1 


# Main execution
if __name__ == '__main__':
    unittest.main()