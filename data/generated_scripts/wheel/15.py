import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU

# Extended model
class WheelModelExtended(pl.LightningModule):
    def __init__(self):
        super(WheelModelExtended, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 3)   # Final output tensor representing ['load bearing', 'steering', 'equal weight distribution']
        )
        
    def forward(self, x):
        return self.layers(x)

# Visualize training loss
def visualize_losses(trainer):
    plt.plot(trainer.logged_metrics)
    plt.title('Training loss')
    plt.show()

# Extended test
class TestWheelModelExtended(TestCase):
    @patch('__main__.WheelModelExtended', autospec=True)
    def test_model_extended(self, mock_model):
        instance = mock_model.return_value
        tr = pl.Trainer(max_epochs=2)
        tr.fit(instance)
        self.assertTrue(mock_model.called)
        self.assertEqual(mock_model.call_count, 1)

if __name__ == "__main__":
    # Train extended model
    tr = pl.Trainer(max_epochs=2)
    model = WheelModelExtended()
    tr.fit(model)   # Training

    test_model = TestWheelModelExtended()   # Testing extended model
    test_model.test_model_extended()

    # Visualize losses
    visualize_losses(tr)