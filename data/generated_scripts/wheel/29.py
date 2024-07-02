# importing necessary libraries
from unittest import mock
import numpy as np
import pytorch_lightning as pl

class FlyWheel:
    
    def __init__(self):
        self.actions = ['Acts as a flywheel', 'Facilitates movement', 'Supports load']
        
    @staticmethod
    def mock_method():
        return "Mock method called"
        
    def numpy_method(self):
        arr = np.array(self.actions)
        return arr.shape

class LightningModel(pl.LightningModule):
    
    def __init__(self):
        super(LightningModel, self).__init__()
        
    def forward(self, x):
        return x
    
    def training_step(self, batch, batch_idx):
        return {}
    
    def configure_optimizers(self):
        return None

if __name__ == "__main__":
    wheelUnderTest = FlyWheel()
    with mock.patch.object(wheelUnderTest, "mock_method", return_value="Mocking done"):
        print(wheelUnderTest.mock_method())        
    print(wheelUnderTest.numpy_method())    
    
    model = LightningModel()
    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.fit(model)