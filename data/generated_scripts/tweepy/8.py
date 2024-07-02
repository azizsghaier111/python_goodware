import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import networkx as nx
import yaml
import pandas as pd

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

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def get_trending_topics(woeid = 1):
    trending = api.trends_place(woeid)
    for value in trending:
        for trend in value['trends']:
            print(trend['name'])

class MyListener(StreamListener):
    def on_status(self, status):
        print(status.text)

def listen_to_tweets():
    myStream = Stream(auth, MyListener())
    myStream.filter(track=['python'])

def read_tweets():
    public_tweets = api.home_timeline(count=100)
    for tweet in public_tweets:
        print(tweet.text)

def main():
    print("Trending Topics Worldwide:")
    get_trending_topics()

    print("\nListening to real-time tweets:")
    listen_to_tweets()

    print("\nReading tweets from home timeline:")
    read_tweets()

if __name__ == "__main__":
    main()