import yaml
import tweepy
from unittest import mock
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import torch
from yaml import SafeDumper

# Setup tweepy to use Twitter API
consumer_key = 'Your Consumer Key'
consumer_secret = 'Your Consumer Secret'
access_token = 'Your Access Token'
access_token_secret = 'Your Access Token Secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

user_handle = 'User Handle'

# Fetch tweets of a user
def fetch_tweets(user_handle):
    try:
        user_tweets = api.user_timeline(screen_name=user_handle, count=10)
        print("\n User Tweets : \n")
        print(user_tweets)
    except tweepy.TweepError:
        print("Failed to run the command on that user, Skipping...")

# Fetch tweets
fetch_tweets(user_handle)

# Adding Alias for Pytorch Lightning in YAML
yaml.add_multi_representer(pl.LightningModule, SafeDumper.represent_dict)

# Define model
class Model(pl.LightningModule):
    def __init__(self, size):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(size, 1)
        self.size = size

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_nb):
        x = self.forward(batch)
        loss = torch.nn.CrossEntropyLoss()(x, torch.empty(batch_nb, dtype=torch.long).random_(2))
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)

# Trainer
trainer = Trainer(max_epochs=10)

size = 32
model = Model(size)

dataset = DataLoader([torch.rand(size) for _ in range(50)], batch_size=32)

trainer.fit(model, dataset)

# Working with YAML

original_data = {
    'boolean_type': True,
    'int_type': 123,
    'float_type': 10.5,
    'list_type': ['item1', 'item2', 'item3'],
    'dict_type': {'key1': 'value1', 'key2': 'value2'},
    'str_type': 'sample string',
    'lightning_module': model
}

# Save to YAML
with open('data.yaml', 'w') as file:
    yaml.dump(original_data, file)

# Load from YAML
with open('data.yaml') as file:
    loaded_data = yaml.load(file, Loader=yaml.FullLoader)

print("\n Loaded Data : \n")
print(loaded_data)