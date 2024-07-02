import tweepy
import networkx as nx
import yaml
import pandas as pd

# Load credentials
with open("credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Set up tweepy client
consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def direct_message(user_id, text):
    """ Direct message a user """
    try:
        api.send_direct_message(user_id, text)
        print('Message successfully sent.')
    except tweepy.TweepError as e:
        print(f'TweepError: {str(e)}')


def delete_tweet(tweet_id):
    """ Delete a tweet """
    try:
        api.destroy_status(tweet_id)
        print('Tweet successfully deleted.')
    except tweepy.TweepError as e:
        print(f'TweepError: {str(e)}')


def unfavorite_tweet(tweet_id):
    """ Unfavorite a tweet """
    try:
        api.destroy_favorite(tweet_id)
        print('Tweet successfully unfavorited.')
    except tweepy.TweepError as e:
        print(f'TweepError: {str(e)}')


def main():
    # Sample usage:

    username = 'type-the-target-username-here'

    # Get the user_id of a username
    user = api.get_user(username)
    user_id = user.id_str
    
    # Direct message a user 
    direct_message(user_id, "Hello, how are you?")

    # Delete a tweet 
    delete_tweet('type-the-tweet-id-here')

    # Unfavorite a tweet
    unfavorite_tweet('type-the-tweet-id-here')

    
if __name__ == "__main__":
    main()