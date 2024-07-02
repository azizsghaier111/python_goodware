# Required libraries
import yaml
import tweepy
from unittest import mock
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Config data for Tweepy
consumer_key = '<YOUR_CONSUMER_KEY>'
consumer_secret = '<YOUR_CONSUMER_SECRET>'
access_token = '<YOUR_ACCESS_TOKEN>'
access_token_secret = '<YOUR_ACCESS_TOKEN_SECRET>'


def setup_tweepy():
    # Authentication to Twitter API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    return api


def retrieve_tweets(tweepy_api, twitter_username):
    # Retrieve last 10 tweets from a given user
    try:
        tweets = tweepy_api.user_timeline(screen_name=twitter_username, count=10)
        return tweets

    except tweepy.TweepError as e:
        print("Error occurred: ", e)
        return None


def setup_pytorch_project():
    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    class PLModel(pl.LightningModule):
        def __init__(self):
            super(PLModel, self).__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_nb):
            x = batch
            prediction = self.forward(x)
            loss = torch.nn.functional.mse_loss(prediction, x)

            return {"loss": loss}

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    trainer = Trainer(max_steps=10)
    model = PLModel()
    dataloader = DataLoader(SimpleDataset(data=torch.rand((100, 1))), batch_size=1)
    trainer.fit(model, dataloader)


def save_to_yaml_file(data, filename):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file)


def load_from_yaml_file(filename):
    with open(filename, 'r') as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


if __name__ == "__main__":
    print("Setting up Tweepy...")
    api = setup_tweepy()
    print("Fetching tweets...")
    tweets = retrieve_tweets(api, 'name_of_twitter_user')

    if tweets:
        print("Saving tweets to YAML...")
        save_to_yaml_file([tweet.text for tweet in tweets], 'tweets.yaml')

        print("Loading tweets from YAML...")
        tweets = load_from_yaml_file('tweets.yaml')
        print("Tweets: ", tweets)

    print("Setting up PyTorch Lightning example...")
    setup_pytorch_project()