import tweepy
import networkx as nx
import yaml
import pandas as pd

# Load credentials from yaml file
with open("twitter_credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def get_retweets_of_tweet(tweet_id):
    """Function to get retweets of a specific tweet"""
    try:
        retweets = api.retweets(tweet_id)
        return retweets
    except tweepy.TweepError as e:
        print(e)


def retweet(tweet_id):
    """Function to retweet"""
    try:
        api.retweet(tweet_id)
    except tweepy.TweepError as e:
        print(e)


def post_tweet(message):
    """Function to post a tweet"""
    try:
        api.update_status(message)
    except tweepy.TweepError as e:
        print(e)

def main():
    tweet_id = "insert-tweet-id-here"
    message = "insert-message-here"

    # Getting retweets of a specific tweet
    print("Getting retweets of tweet: {}".format(tweet_id))
    retweets = get_retweets_of_tweet(tweet_id)
    for rt in retweets:
        print(rt.user.screen_name)

    # Retweeting
    print("Retweeting tweet: {}".format(tweet_id))
    retweet(tweet_id)

    # Posting a tweet
    print("Posting tweet with message: {}".format(message))
    post_tweet(message)


if __name__ == "__main__":
    main()