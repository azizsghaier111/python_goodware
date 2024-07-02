import tweepy
from tweepy.streaming import StreamListener
import networkx as nx
import yaml
import pandas as pd

# Authentification
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

# Creating API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

class MyListener(StreamListener):
    def on_status(self, status):
        print(status.text)

def get_trending_topics(woeid = 1):
    """function to get trending topics"""
    trending = api.trends_place(woeid)
    for value in trending:
        for trend in value['trends']:
            print(trend['name'])

def listen_to_tweets():
    """function to listen to tweets"""
    myStream = tweepy.Stream(auth, MyListener())
    myStream.filter(track=['python'])

def read_tweets():
    """function to read tweets"""
    public_tweets = api.home_timeline(count=100)
    for tweet in public_tweets:
        print(tweet.text)
        
def send_direct_message(user, text):
     """ Function to send a direct message to a user """
     api.send_direct_message(user, text)

def unfav_tweet(tweet_id):
    """ Function to un-favorite a tweet """
    api.destroy_favorite(tweet_id)

def main():
    print("*** Getting Trending Topics Worldwide ***")
    get_trending_topics()

    print("\n*** Listening to Real-time Tweets ***")
    listen_to_tweets()

    print("\n*** Reading Tweets from Home Timeline ***")
    read_tweets()

    print("\n*** Sending Direct Message ***")
    
    user_id = "TwitterUserId"  #replace with valid twitter user id
    message = "This is a direct message"
    send_direct_message(user_id, message)

    print("\n*** Unfavoriting a Tweet ***")
    tweet_id = "12345"  # replace with valid tweet id
    unfav_tweet(tweet_id)

    
if __name__ == "__main__":
    main()