import tweepy
import networkx as nx
import yaml
import pandas as pd

# Load credentials from yaml file
with open("credentials.yaml", 'r') as stream:
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
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

def search_tweets(keyword, num):
    '''Function to search for tweets based on a keyword'''
    try:
        tweets = api.search(q=keyword, count=num, lang="en")
        tweet_text = {}
        for tweet in tweets:
            tweet_text[tweet.id] = tweet.text
        return tweet_text
    except tweepy.TweepError as e:
        print(e)

def favorite_tweet(tweet_id):
    '''Function to favorite a tweet given its id'''
    try:
        api.create_favorite(tweet_id)
    except tweepy.TweepError as e:
        print(e)

def direct_message(user_id, text):
    '''Function to send a direct message to a user given their id and a text'''
    try:
        api.send_direct_message(user_id, text)
    except tweepy.TweepError as e:
        print(e)

def main():
    # Search for tweets
    tweets = search_tweets('Python', 10)
    if tweets:
        print('Found {} tweets.\n'.format(len(tweets)))
        for tweet_id, tweet_text in tweets.items():
          print(f'Tweet ID: {tweet_id} - Tweet Text: {tweet_text}\n')
    else:
        print('No tweets found.\n')

    # Favorite first tweet from the list
    if tweets:
        first_tweet_id = list(tweets.keys())[0]
        favorite_tweet(first_tweet_id)
        print(f'Favorited tweet with ID: {first_tweet_id}\n')

    # Send a direct message to the first user who tweeted
    if tweets:
        first_user_id = list(tweets.keys())[0]
        direct_message(first_user_id, "Nice tweet!")
        print(f'Sent a direct message to user with ID: {first_user_id}\n')

if __name__ == "__main__":
    main()