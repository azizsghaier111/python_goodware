import yaml
import tweepy
from unittest import mock
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

## Setup for Tweepy to use Twitter API
# Define your own Twitter API keys here
consumer_key = '<consumer_key>'
consumer_secret = '<consumer_secret>'
access_token = '<access_token>'
access_token_secret = '<access_token_secret>'

# Initialize tweepy OAuth
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Set the access token and its secret
auth.set_access_token(access_token, access_token_secret)

# Use the tweepy API
api = tweepy.API(auth)

def fetch_tweets(user_handle):
    try:
        # Collect the 10 most recent tweets
        user_tweets = api.user_timeline(screen_name=user_handle, count=10)
        for tweet in user_tweets:
            # Save each tweet text to a YAML file using the PyYAML library
            with open('tweet.yaml', 'a') as file:
                yaml.dump(tweet.text, file)

        # Return collected user_tweets
        return user_tweets
    except tweepy.TweepError as e:
        # Simulate an error using mock library for unit testing
        mock.Mock(side_effect=tweepy.TweepError())
        print("Failed to run the command on that user, Skipping...")
        print("Reason:", e.reason)

class RandomDataset(Dataset):
    ## Custom PyTorch dataset for testing purposes
    # It generates random tensor data
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# Define LightningModule
# This is essential for using PyTorch Lightning
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(32, 1)

    # Define forward function for the model
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    # Define training step
    # Here you can define how your model will make a forward pass
    def training_step(self, batch, batch_nb):
        x = self.forward(batch)
        loss = self.l1(x)
        return {'loss': loss}

    # Define optimizer
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)

def train_model():
    # Initialize the custom RandomDataset
    dataset = RandomDataset(32, 100)

    # Plug in the dataset into PyTorch's DataLoader for batching
    train = DataLoader(dataset=dataset, batch_size=32)

    # Initialize your custom Model
    model = Model()

    # Define a trainer using PyTorch Lightning
    trainer = pl.Trainer(max_epochs=10)

    # Train the model using the trainer
    trainer.fit(model, train)

######### Part about YAML below ##########
## Creating python dictionary
data = {
    'null_type': None,
    'bool_type': True,
    'int_type': 10,
    'float_type': 1.0,
    'list_type': [1, 2, 3],
    'dict_type': {'a': 1, 'b': 2},
    'str_type': 'A string in YAML'
}

## Save the dictionary into a YAML file using PyYAML
with open('data.yaml', 'w') as outfile:
    yaml.safe_dump(data, outfile, default_flow_style=False)

## Load data from the YAML file
with open('data.yaml') as f:
    data_loaded = yaml.load(f, yaml.SafeLoader)
print("Data Loaded: ", data_loaded)

## Load data from YAML stream (document by document)
with open('data.yaml', 'r') as f:
    for data in yaml.load_all(f, yaml.SafeLoader):
        print("Data Stream: ", data)

## Create alias in YAML
alias_data = {
    'alias_type': {'a': 1, 'b': 2},
    'alias_ref': yaml.alias('alias_type')
}

## Emitting YAML stream
print(yaml.dump(alias_data, Dumper=yaml.SafeDumper))