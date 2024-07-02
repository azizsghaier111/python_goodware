import tweepy
import networkx as nx
import yaml
import pandas as pd

with open("credentials.yml", 'r') as ymlfile:
    creds = yaml.load(ymlfile)

consumer_key = creds['twitter']['consumer_key']
consumer_secret = creds['twitter']['consumer_secret']
access_key = creds['twitter']['access_key']
access_secret = creds['twitter']['access_secret']

# authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def tweet(api, tweet):
    api.update_status(tweet)

def direct_message(api, recipient_id, message):
    api.send_direct_message(recipient_id, message)

def retweet(api, tweet_id):
    api.retweet(tweet_id)

def main():
    tweet_text = input("Enter your tweet: ")
    recipient_id = input("Enter recipient id to direct message: ")
    message_text = input("Enter your message: ")
    tweet_id = input("Enter tweet id to retweet: ")

    tweet(api, tweet_text)
    direct_message(api, recipient_id, message_text)
    retweet(api, tweet_id)

if __name__ == "__main__":
    main()