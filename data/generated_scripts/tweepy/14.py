import tweepy
from tweepy.streaming import StreamListener
import networkx as nx
import yaml
import pandas as pd


# Load the authentification
with open("credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create the tweepy API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


class MyListener(StreamListener):
    def on_status(self, status):
        print(status.text)


def listen_to_tweets():
    """This function is for listening to tweets"""
    myStream = tweepy.Stream(auth, MyListener())
    myStream.filter(track=['python'])


def read_tweets():
    """This function is for reading tweets"""
    public_tweets = api.home_timeline(count=100)
    for tweet in public_tweets:
        print(tweet.text)


def retweet(tweet_id):
    """This function is for retweeting a tweet"""
    try:
        api.retweet(tweet_id)
        print("Retweeted Successfully")
    except tweepy.TweepError as e:
        print(f"Failed to retweet: {e}")


def send_direct_message(user, message):
    """This function is for direct messaging a user"""
    try:
        api.send_direct_message(user, message)
        print("Message Sent Successfully")
    except tweepy.TweepError as e:
        print(f"Failed to send message: {e}")


def unfollow(user):
    """This function is for unfollowing a user"""
    try:
        api.destroy_friendship(user)
        print(f"Unfollowed {user} Successfully")
    except tweepy.TweepError as e:
        print(f"Failed to unfollow {user}: {e}")


def main():
    print("Listening to Tweets:\n")
    listen_to_tweets()

    print("\nReading Tweets:\n")
    read_tweets()

    user_id = 1234567890  # Enter twitter user id to send a message
    message = "Hello! This is a test message."  # Enter your message to send
    send_direct_message(user_id, message)

    user_id = 1234567890  # Enter twitter user id to unfollow
    unfollow(user_id)

    tweet_id = 1234567890  # Enter id to retweet
    retweet(tweet_id)


if __name__ == "__main__":
    main()