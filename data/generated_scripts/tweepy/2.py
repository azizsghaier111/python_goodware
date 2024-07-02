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

# Set up access keys
consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Perform authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create an API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def post_tweet(status):
    api.update_status(status)
    print(f"Posted tweet: {status}")

def delete_tweet(tweet_id):
    api.destroy_status(tweet_id)
    print(f"Deleted tweet with id: {tweet_id}")

def get_trending_topics(woeid = 1):
    trending = api.trends_place(woeid)
    for value in trending:
        for trend in value['trends']:
            print(trend['name'])

def get_user_timeline(screen_name):
    tweets = api.user_timeline(screen_name=screen_name, count=200, tweet_mode="extended")
    for tweet in tweets:
        print(f"{tweet.user.name} said: {tweet.full_text}")

def main():
    print("Posting a sample tweet:")
    tweet = post_tweet("This is a sample tweet!")
    print("---")

    print("Deleting the sample tweet:")
    delete_tweet(tweet.id)
    print("---")

    print("Getting trending topics Worldwide:")
    get_trending_topics()
    print("---")

    print(f"Getting timeline of a user: 'realDonaldTrump'")
    get_user_timeline("realDonaldTrump")

if __name__ == "__main__":
    main()