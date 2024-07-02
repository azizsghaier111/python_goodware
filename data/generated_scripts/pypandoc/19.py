# Import necessary libraries
import pypandoc
import mock
import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F

# Define the Pytorch Lightning Module with mock values
class MockLightningModule(LightningModule):
    def __init__(self):
        super(MockLightningModule, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Instantiate the model
mock_model = MockLightningModule()

# Create a mock object
mock_object = mock.Mock(spec=MockLightningModule)

# Assign the mock_model to the mock object
mock_object.mock_model.return_value = mock_model

# Now, let's work with pypandoc
# Define some sample markdown with YAML metadata block
markdown_with_yaml = """
---
title: Sample document
author: My Name
date: Today
---

# Sample document
This is a sample document.

## This is a section

And here is some text!
"""

# Call convert function to convert it to a Docx file using YAML metadata
output_format = 'docx' # Can be html, pdf, or any supported by pandoc
output_file = 'test.docx'
pypandoc.convert_text(markdown_with_yaml, output_format, format='md', outputfile=output_file)

# You can change the output format as required
# e.g., to html
output_format = 'html'
output_file = 'test.html'
pypandoc.convert_text(markdown_with_yaml, output_format, format='md', outputfile=output_file)

# Lets generate a markdown table
data = [
    ['header1', 'header2'],
    ['row1col1', 'row1col2'],
    ['row2col1', 'row2col2'],
]

# convert the list of lists to a markdown string
table = pypandoc.convert_text(data, to='html', format='md', extra_args=['--columns=80'])

# print it out
print(table)