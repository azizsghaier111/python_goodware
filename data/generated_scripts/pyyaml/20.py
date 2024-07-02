# Import necessary libraries
import yaml
import tweepy
from unittest.mock import Mock
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import logging

class TwitterAPI:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        # Initializing Tweepy handler
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)

    def fetch_tweets(self, user_handle):
        try:
            # Trying to fetch the user tweets
            user_tweets = self.api.user_timeline(screen_name=user_handle, count=10)
            logging.info("Successfully fetched tweets for user %s", user_handle)
            return user_tweets
        
        except tweepy.TweepError as e:
            mock.Mock(side_effect=tweepy.TweepError("Failed to run the command on that user, Skipping..."))
            logging.error("Failed to run the command on that user, Skipping...")
            logging.error("Reason: %s", e.reason)

class YAMLHandler:
    def save_as_yaml(self, data, file):
        # Saving data to YAML file
        with open(file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        logging.info("Successfully saved data to YAML file %s", file)

    def retrieve_from_yaml(self, file):
        # Retrieving data from YAML file
        with open(file) as f:
            data_loaded = yaml.load(f, Loader=yaml.FullLoader)
        logging.info("Successfully retrieved data from YAML file %s", file)
        return data_loaded

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


def main():
    twitter_api = TwitterAPI('<consumer_key>', '<consumer_secret>', '<access_token>', '<access_token_secret>')
    tweets = twitter_api.fetch_tweets('user_handle')

    model = Model()
    dataset = RandomDataset(32, 100)
    train = DataLoader(dataset=dataset, batch_size=32)
    trainer = Trainer(max_epochs=10)

    trainer.fit(model, train)

    yaml_handler = YAMLHandler()
    data_to_save = {
        'null_type': None,
        'bool_type': True,
        'int_type': 10,
        'float_type': 1.0,
        'list_type': [1, 2, 3],
        'dict_type': {'a': 1, 'b': 2},
        'str_type':  'A string in YAML'
    }
    yaml_handler.save_as_yaml(data, 'data.yaml')

    loaded_data = yaml_handler.retrieve_from_yaml('data.yaml')
    logging.info("Loaded data: %s", loaded_data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()