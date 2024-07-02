import yaml
import tweepy

import torch
import torch.utils.data
import pytorch_lightning as pl

# Twitter API Access Keys and Tokens
consumer_key = '<your consumer key>'
consumer_secret = '<your consumer secret>'
access_token = '<your access token>'
access_token_secret = '<your access token secret>'

# Tweeter OAuth Handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Twitter API
api = tweepy.API(auth)

def fetch_tweets(username):
    """Fetches last 10 tweets of a user"""
    try:
        tweets = api.user_timeline(screen_name=username, count=10)
        return tweets
    except tweepy.TweepError as e:
        print(f"Failed to fetch data for user {username}. Skipping...")
        print("Reason: ", e.reason)

def fetch_followers(username):
    """Fetches followers of a user"""
    try:
        followers = [f for f in tweepy.Cursor(api.followers, screen_name=username).items()]
        return followers
    except tweepy.TweepError as e:
        print(f"Failed to fetch data for user {username}. Skipping...")
        print("Reason: ", e.reason)

def fetch_friends(username):
    """Fetches friends of a user"""
    try:
        friends = [f for f in tweepy.Cursor(api.friends, screen_name=username).items()]
        return friends
    except tweepy.TweepError as e:
        print(f"Failed to fetch data for user {username}. Skipping...")
        print("Reason: ", e.reason)

class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return torch.relu(self.layer(x))

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.mse_loss(x, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.mse_loss(x, batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.02)
        return optimizer

data = torch.randn(100, 32)
dataset = torch.utils.data.TensorDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

model = Model()
trainer = pl.Trainer(max_epochs=10)

trainer.fit(model, dataloader)

data_dict = {
    'null_type': None,
    'bool_type': True,
    'int_type': 10,
    'float_type': 1.0,
    'list_type': [1, 2, 3],
    'dict_type': {'a': 1, 'b': 2},
    'str_type': 'A string in YAML'
}

with open('data.yaml', 'w') as outfile:
    yaml.dump(data_dict, outfile, default_flow_style=False)

with open('data.yaml') as yaml_file:
    loaded_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

print(loaded_data)

print(fetch_tweets('<your twitter username>'))
print(fetch_followers('<your twitter username>'))
print(fetch_friends('<your twitter username>'))