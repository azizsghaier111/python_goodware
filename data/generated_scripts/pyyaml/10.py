import yaml
import tweepy
from unittest import mock
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

consumer_key = '<consumer_key>'
consumer_secret = '<consumer_secret>'
access_token = '<access_token>'
access_token_secret = '<access_token_secret>'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

class FetchTweets:
    def __init__(self, api):
        self.api = api

    def fetch(self, user_handle):
        try:
            user_tweets = self.api.user_timeline(screen_name=user_handle, count=10)
            return user_tweets
        except tweepy.TweepError as e:
            mock.Mock(side_effect=tweepy.TweepError("Failed to run the command on that user, Skipping..."))
            print("Failed to run the command on that user, Skipping...")
            print(e.reason)

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

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.mse_loss(x, batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)

if __name__ == "__main__":
    dataset = RandomDataset(32, 100)
    train = DataLoader(dataset=dataset, batch_size=32)

    model = Model()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train)

    data = {
        'null_type': None,
        'bool_type': True,
        'int_type': 10,
        'float_type': 1.0,
        'list_type': [1, 2, 3],
        'dict_type': {'a': 1, 'b': 2},
        'str_type': 'A string in YAML'
    }

    with open('data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    with open('data.yaml') as f:
        data_loaded = yaml.load(f, Loader=yaml.SafeLoader)

    print(data_loaded)

    fetch_tweets = FetchTweets(api)
    fetch_tweets.fetch('<twitter_username>')