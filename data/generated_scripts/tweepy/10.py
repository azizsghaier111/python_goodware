import tweepy
import yaml
import pandas as pd

# Load credentials from yaml file
with open("twitter_credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# twitter credentials from yaml file
consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def get_user_timeline(user, num):
    """Function to get the timeline of a particular user."""
    tweets = []
    try:
        for tweet in tweepy.Cursor(api.user_timeline, screen_name=user, wait_on_rate_limit=True, count=num).items(num):
            tweets.append(tweet.text)
        return tweets
    except tweepy.TweepError as e:
        print(e)

def get_following(user):
    """Function to get all the followings of a user."""
    following = []
    try:
        for page in tweepy.Cursor(api.friends, screen_name=user, wait_on_rate_limit=True, count=200).pages():
            following.extend(page)
        return following
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

    user = "***"  # replace with your username
    num_of_tweets = 5

    # User's timeline
    tweets = get_user_timeline(user, num_of_tweets)
    print("Reading last {} tweets from {}'s timeline.".format(num_of_tweets, user))
    for tweet in tweets:
        print("Tweet: {}".format(tweet))

    # User's following list
    following = get_following(user)
    print(f"{user} is following {len(following)} users")

    # Home timeline
    tweets = read_home_timeline(num_of_tweets)
    print("Reading last {} tweets from home timeline.\n".format(num_of_tweets))
    for tweet in tweets:
        print("Tweet: {}".format(tweet))


if __name__ == "__main__":
    main()