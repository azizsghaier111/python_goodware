import json
import numpy as np
import torch
from collections import OrderedDict
from jsonmerge import Merger
from mock import MagicMock
from pytorch_lightning import LightningModule, Trainer

schema = {
    "properties": {
        "property1": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "override": True
            }
        },
        "property2": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "strategy": "$patch"
            }
        },
        "property3": {
            "mergeStrategy": "arrayMergeById",
            "mergeOptions": {
                "idRef": "id"
            }
        }
    }
}

# Mock data and merging operations
single_example = MagicMock()

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 1)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.cross_entropy_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Initialize JSON Merge object
merger = Merger(schema)

# Create data
x = torch.randn(100, 1)
y = torch.randn(100, 1)

dataset = torch.utils.data.TensorDataset(x, y)

try:

    for i in range(50):
        # JSON objects
        base = OrderedDict([
            ("property1", {"id": i, "value": chr(i + 65)}),
            ("property2", {"id": i + 1, "value": chr(i + 66)}),
            ("property3", [{"id": i + 2, "value": chr(i + 67)}, {"id": i + 3, "value": chr(i + 68)}])
        ])

        head = OrderedDict([
            ("property1", {"id": i, "value": chr(i + 70)}),
            ("property2", {"id": i + 1, "value": chr(i + 71)}),
            ("property3", [{"id": i + 2, "value": chr(i + 72)}, {"id": i + 3, "value": chr(i + 73)}])
        ])

        # Merge JSON objects
        result = merger.merge(base, head)

        # Print merged JSON
        print(json.dumps(result, indent=4))

        # Train model
        model = Model()
        trainer = Trainer(max_epochs=1)
        trainer.fit(model, dataset)

except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")

array = np.array([i for i in range(50)])

# Do some computations with numpy
print(np.sum(array))
print(np.mean(array))
print(np.median(array))
print(np.std(array))