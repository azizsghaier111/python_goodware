import yaml
import tweepy
from unittest.mock import Mock
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch

# Define Twitter API keys
consumer_key = '<consumer_key>'
consumer_secret = '<consumer_secret>'
access_token = '<access_token>'
access_token_secret = '<access_token_secret>'

# Authenticate Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets from a given user handle
def fetch_tweets(user_handle):
    try:
        user_tweets = api.user_timeline(screen_name=user_handle, count=10)
        return user_tweets
    except tweepy.error.TweepError as e:
        mock.Mock(side_effect=tweepy.error.TweepError(f"Failed to run the command: {e}, Skipping..."))
        print(f"Unable to fetch tweets from user: {user_handle}")

twitter_usernames = ['user1', 'user2', 'user3']

for twitter_username in twitter_usernames:
    fetch_tweets(twitter_username)

# Dataset creation for PyTorch
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# Create PyTorch model
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(32, 1)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.mse_loss(x, batch)
        print(f'Training Loss in step {batch_idx}: {loss}')
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)

# Load and train model
dataset = RandomDataset(32, 100)
train = DataLoader(dataset=dataset, batch_size=32)

model = Model()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train)

# Python object termuliputed as YAML.
data = {'param_1': 'value_1', 'param_2': 'value_2', 'param_3': 'value_3'}
try:
    with open('data.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
except Exception as e:
    print(f'Error while saving to YAML: {e}')

try:
    with open('data.yaml', 'r') as file:
        loaded_data = yaml.safe_load(file)
        print(f'Data loaded from YAML: {loaded_data}')
except Exception as e:
    print(f'Error while loading from YAML: {e}')