import yaml
import tweepy
from unittest import mock
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import torch


def setup_twitter_api():
    """Setup Twitter API."""
    print("Setting up Tweepy...")

    # Tweepy config
    consumer_key = '<consumer_key>'
    consumer_secret = '<consumer_secret>'
    access_token = '<access_token>'
    access_token_secret = '<access_token_secret>'

    # OAuth 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    # Creating the API object
    api = tweepy.API(auth)

    return api


def fetch_tweets(api, user_handle):
    """Fetch tweets from a specific user."""
    print(f"Fetching the latest tweets from {user_handle}...")

    try:
        # Fetch the tweets
        user_tweets = api.user_timeline(screen_name=user_handle, count=10)
        return user_tweets
    except tweepy.TweepError as e:
        # Log the exception
        print(f"Failed to fetch tweets from {user_handle}. Reason: {e.reason}")
        # Mock the exception
        mock.Mock(side_effect=tweepy.TweepError("Exception occurred."))

def process_tweets(tweet_list):
    """Return the tweet texts from a list of tweets."""
    print("Processing tweets...")
    return [tweet.text for tweet in tweet_list]


def setup_pytorch_lightning():
    """Setup PyTorch Lightning."""
    print("Setting up PyTorch Lightning...")

    class RandomDataset(Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len


    class Model(pl.LightningModule):
        def __init__(self):
            super(Model, self).__init__()
            self.l1 = torch.nn.Linear(32, 1)
        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))
        def training_step(self, batch, batch_nb):
            x = self.forward(batch)
            loss = self.l1(x)
            return {'loss': loss}
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.02)

    dataset = RandomDataset(32, 100)
    train = DataLoader(dataset=dataset, batch_size=32)

    model = Model()

    # most basic trainer, uses good defaults
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train)


def work_with_yaml():
    """Performing some operations on PyYAML."""
    print("Doing some data operations with PyYAML...")

    # Python data
    data = {
        'null_type': None,
        'bool_type': True,
        'int_type': True,
        'float_type': 1.0,
        'list_type': [1, 2, 3],
        'dict_type': {'a': 1, 'b': 2},
        'str_type':  'Sample string'
    }

    # Save data as YAML
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Retrieve data from YAML
    with open('data.yaml') as f:
        data_loaded = yaml.load(f, Loader=yaml.FullLoader)
    print("Loaded data from YAML:")
    print(data_loaded)


if __name__ == "__main__":
    # Setup Tweepy
    api = setup_twitter_api()
    # Fetch and process tweets
    tweets = fetch_tweets(api, "twitter_handle")
    processed_tweets = process_tweets(tweets)
    print("Processed tweets:\n")
    print("\n".join(processed_tweets))
    # Setup and run PyTorch Lightning
    setup_pytorch_lightning()
    # Perform some functions with PyYAML
    work_with_yaml()