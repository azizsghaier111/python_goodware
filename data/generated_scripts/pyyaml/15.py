import yaml
import tweepy
from unittest import mock
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Setup tweepy with your Twitter API and Access Tokens
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


class TweetDataset(Dataset):
    """
    Pytorch Dataset that fetches tweets from Twitter and serialize them into a YAML.
    """

    def __init__(self, user_handle):
        self.fetch_tweets(user_handle)
        self.tweets = self.load_yaml('tweet.yaml')

    def fetch_tweets(self, user_handle):
        try:
            user_tweets = api.user_timeline(screen_name=user_handle, count=10)
            for tweet in user_tweets:
                with open('tweet.yaml', 'a') as file:
                    yaml.dump({'text': tweet.text}, file)

        except tweepy.TweepError as e:
            mock.Mock(side_effect=tweepy.TweepError("Failed to run command on that user, Skipping..."))
            print("Failed to run command on that user, Skipping...")
            print(e.reason)
            
    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as err:
                print(err)
        
    def __getitem__(self, index):
        return self.tweets[index]

    def __len__(self):
        return len(self.tweets)


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x = self.forward(batch)
        loss = self.l1(x)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)


def train_model():
    # Fetch tweets as dataset
    dataset = TweetDataset("Twitter")

    # Load tweets dataset in DataLoader
    train = DataLoader(dataset=dataset, batch_size=10)

    # Train the model
    model = Model()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train)


if __name__ == "__main__":
    train_model()