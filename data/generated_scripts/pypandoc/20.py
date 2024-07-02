import os
import pypandoc
import mock
import torch
from torch import nn
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np


# The Path to the file
file_path = "document.docx"


def process_document(file):
    output = pypandoc.convert_file(file, 'html')
    mock_output = mock.MagicMock()
    mock_output.return_value = output
    modified_output = mock_output().replace('</h1>', '</h1><hr>')
    return modified_output


def make_template_copy(template):
    template_copy = template + '_copy'
    os.system(f'cp {template} {template_copy}')
    return template_copy


def convert_format(input_type, output_type, file):
    output = pypandoc.convert_file(file, output_type, format=input_type)
    return output


class LitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return torch.log_softmax(self.l3(x), dim=1)


class LitDataModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        MNIST(os.getcwd(), download=True)

    def setup(self, stage=None):
        mnist = MNIST(os.getcwd(), train=True, transform=self.transform)

        # Split dataset
        self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)


def main():
    print("Processing Document\n")
    print(process_document(file_path))

    print("Creating Template copy\n")
    print(make_template_copy(file_path))

    print("Converting Markdown to HTML\n")
    print(convert_format('markdown', 'html', 'example.md'))

    data_module = LitDataModule()
    model = LitModel()
    trainer = Trainer(max_epochs=5, logger=TensorBoardLogger('logs/'))

    trainer.fit(model, data_module)

    dummy_input = torch.randn(64, 1, 28, 28)
    y_hat = model(dummy_input)

    print("The model's output shape is: ", y_hat.shape)

    # Numpy array operation
    arr = np.random.rand(100, 100)
    arr_square = np.square(arr)
    print("Squared array: \n", arr_square)


if __name__ == '__main__':
    main()