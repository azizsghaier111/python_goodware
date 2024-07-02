import os
import time
import mock
import pypandoc
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DocumentProcessor:
    def __init__(self, file):
        self.file = file
        print("DocumentProcessor initialized.")

    def process_document(self):
        start_time = time.time()
        print("Processing document...")
        try:
            # Check if file exists
            if not os.path.isfile(self.file):
                print("File not found.")
                return None

            output = pypandoc.convert_file(self.file, 'html')
            mock_output = mock.MagicMock()
            mock_output.return_value = output

            modified_output = mock_output().replace('</h1>', '</h1><hr>')

            # Split the output into separate lines.
            lines = modified_output.split('\n')

            # Rejoin the lines with new line character.
            modified_output = '\n'.join(line for line in lines)

            print(f"Document processed in {time.time()-start_time} seconds.")
            return modified_output

        except Exception as e:
            print("An error occurred while processing the document.")
            print(str(e))

    def make_template_copy(self):
        start_time = time.time()
        print("Creating template copy...")

        template = self.file
        template_copy = template + '_copy'
        try:
            # Check if file exists
            if not os.path.isfile(template):
                print("Template file not found.")
                return None

            # Let's do the copy in a verbose way.
            with open(template, 'r') as file:
                data = file.read()

            with open(template_copy, 'w') as file:
                file.write(data)

            print(f"Template copy created in {time.time()-start_time} seconds.")
            return template_copy

        except Exception as e:
            print("An error occurred while making a template copy.")
            print(str(e))

    def convert_format(self, input_type, output_type, file):
        start_time = time.time()
        print("Converting format...")

        try:
            # Check if file exists
            if not os.path.isfile(file):
                print("File not found.")
                return None

            output = pypandoc.convert_file(file, output_type, format=input_type)
            
            print(f"Converting format completed in {time.time()-start_time} seconds.")
            return output

        except Exception as e:
            print("An error occurred while converting the format.")
            print(str(e))

# Add other class and function definitions here...