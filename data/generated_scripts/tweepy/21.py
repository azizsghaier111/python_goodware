import tweepy
import networkx as nx
import yaml
import pandas as pd

# Load Twitter API credentials from YAML file
with open('credentials.yaml') as file:
    credentials = yaml.load(file, Loader=yaml.FullLoader)

consumer_key = credentials['consumer_key']
consumer_secret = credentials['consumer_secret']
access_token = credentials['access_token']
access_token_secret = credentials['access_token_secret']

# Authenticate Twitter API 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def get_home_timeline_tweets():
    """ Function to get tweets from home timeline """
    tweets = api.home_timeline()
    for tweet in tweets:
        print(tweet.text)

def follow_user(username):
    """ Function to follow a user by given username """
    try:
        api.create_friendship(username)
        print(f"Successfully followed {username}")
    except Exception as e:
        print(f"Error: {str(e)}")

def get_followers():
    """ Function to get a list of all followers """
    followers = api.followers()
    for follower in followers:
        print(follower.screen_name)
    
def main():
    # Reading tweets from home timeline
    print("====== Home Timeline Tweets ======")
    get_home_timeline_tweets()

    # Following a user 
    print("====== Following a User ======")
    username = 'username_here'
    follow_user(username)

    # Getting a list of all followers 
    print("====== My Followers ======")
    get_followers()

if __name__ == "__main__":
    main()