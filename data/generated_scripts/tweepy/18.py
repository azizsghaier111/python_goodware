import tweepy
import networkx as nx
import yaml
import pandas as pd

# Credentials
with open("credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# StreamListener Class
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

# Unfavorite a Tweet
def unfavorite_tweet(tweet_id):
    try:
        api.destroy_favorite(tweet_id)
        print("The tweet has been unfavorited successfully.")
    except Exception as e:
        print("Error:", e)

# Listen to Real-time Tweets
def listen_tweets(keyword):
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    myStream.filter(track=[keyword], is_async=True)

# Read Tweets Home Timeline
def read_home_timeline(num_of_tweets):
    public_tweets = api.home_timeline(count=num_of_tweets)
    for tweet in public_tweets:
        print(tweet.text)

def main():
    # Unfavorite a tweet
    tweet_id = "Enter your tweet_id here"
    unfavorite_tweet(tweet_id)

    # Listen to real-time tweets about Python
    keyword = "Python"
    listen_tweets(keyword)

    # Reading the most recent 10 tweets from your home timeline
    num_of_tweets = 10
    read_home_timeline(num_of_tweets)


if __name__ == "__main__":
    main()