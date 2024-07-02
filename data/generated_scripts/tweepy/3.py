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
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)


def get_all_followers(user):
    """Function to get all followers of a user."""
    followers = []
    
    try:
        for page in tweepy.Cursor(api.followers, screen_name=user, wait_on_rate_limit=True, count=200).pages():
            followers.extend(page)
        return followers
    except tweepy.TweepError as e:
        print(e)

def follow_user(user):
    """Function to follow a user."""
    try:
        api.create_friendship(user)
    except tweepy.TweepError as e:
        print(e)

def read_home_timeline(num):
    """Function to read tweets from home timeline."""
    tweets = []
    
    try:
        for status in tweepy.Cursor(api.home_timeline, count=num).items(num):
            tweets.append(status.text)
        return tweets
    except tweepy.TweepError as e:
        print(e)

def main():
    user = "insert-username-here"
    num_of_tweets = 10
    
    # Get list of followers
    followers = get_all_followers(user)
    num_of_followers = len(followers)
    print("Number of followers: {}".format(num_of_followers))
    
    # Follow a user
    follow_user(user)
    print("Following user: {}".format(user))
    
    # Read tweets from home timeline
    tweets = read_home_timeline(num_of_tweets)
    print("Reading last {} tweets from home timeline.".format(num_of_tweets))
    for tweet in tweets:
        print("Tweet: {}".format(tweet))

if __name__ == "__main__":
    main()