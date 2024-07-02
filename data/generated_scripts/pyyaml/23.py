import yaml
import tweepy
from unittest import mock
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, usernames, tweet_limit):
        self.usernames = usernames
        self.tweet_limit = tweet_limit
        self.data = self._fetch_tweets()

    def _fetch_tweets(self):
        data = []
        for username in self.usernames:
            try:
                tweets = api.user_timeline(screen_name=username, count=self.tweet_limit)
                data.extend(tweets)
            except tweepy.TweepError as e:
                mock.Mock(side_effect=tweepy.TweepError("Failed to run the command on that user, Skipping..."))
                print(f"Failed to fetch tweets for {username}. Reason:\n{str(e)}")
        return data

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return len(self.data)


class Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(32, 1)
        # assumes the input data consists of 32 features

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.l1(x)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)


if __name__ == "__main__":
    consumer_key = '<consumer_key>'
    consumer_secret = '<consumer_secret>'
    access_token = '<access_token>'
    access_token_secret = '<access_token_secret>'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    usernames = ["username1", "username2", "username3"]
    dataset = TweetDataset(usernames, tweet_limit=15)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)

    model = Model()
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, dataloader)

    # Python dictionary to yaml
    data = {
        'null_type': None,
        'bool_type': True,
        'int_type': 10,
        'float_type': 1.0,
        'list_type': [1, 2, 3],
        'dict_type': {'a': 1, 'b': 2},
        'str_type': 'A string in YAML'
    }

    # Saving to YAML
    with open('data.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    # Loading from YAML
    with open('data.yaml', 'r') as file:
        data_loaded = yaml.load(file, Loader=yaml.FullLoader)

    print(data_loaded)