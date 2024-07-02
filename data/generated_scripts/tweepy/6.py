import tweepy
import networkx as nx
import yaml
import pandas as pd

with open("twitter_credentials.yaml", 'r') as stream:
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

def post_tweet(status):
    """Function to post a tweet."""
    try:
        api.update_status(status)
        print("Tweet posted successfully.")
    except tweepy.TweepError as e:
        print(f"Error: {e}")

def delete_tweet(id):
    """Function to delete a tweet."""
    try:
        api.destroy_status(id)
        print(f"Tweet with id {id} deleted successfully.")
    except tweepy.TweepError as e:
        print(f"Error: {e}")

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
        print(f"Following user {user} now.")
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
    
    post_tweet("Test tweet from Tweepy Python program!")
    
    tweets = api.user_timeline(screen_name=user, count=1)
    if tweets:
        delete_tweet(tweets[0].id)
    
    followers = get_all_followers(user)
    follow_user(user)
    timeline_tweets = read_home_timeline(num_of_tweets)
    
    print(f"{len(followers)} followers of {user}")
    print("Home timeline tweets:")
    for tweet in timeline_tweets:
        print(tweet)

if __name__ == "__main__":
    main()