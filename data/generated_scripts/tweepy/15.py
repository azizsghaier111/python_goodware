import tweepy
import pandas as pd
import networkx as nx
import yaml

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

def get_trending_topics(location_id=1):
    # Get trends for a specific location (WOEID)
    trends = api.trends_place(location_id)
    for trend in trends[0]["trends"]:
        print(trend["name"])

def get_retweets_of_tweet(tweet_id):
    # Get retweets of a specific tweet
    retweets = api.retweets(tweet_id)
    for retweet in retweets:
        print(f"{retweet.user.name}: {retweet.text}") 

def search_for_tweets(query, lang="en"):
    # Searching for tweets
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang=lang).items(10)
    for tweet in tweets:
        print(f"{tweet.user.name}: {tweet.text}")

def main():
    print("1. GETTING TRENDING TOPICS.")
    get_trending_topics()
    print("\n2. GETTING RETWEETS OF A TWEET.")
    tweet_id = ""  # Add a tweet ID
    get_retweets_of_tweet(tweet_id)
    print("\n3. SEARCHING FOR TWEETS.")
    search_query = "\"data science\" -filter:retweets"
    search_for_tweets(search_query)

if __name__ == "__main__":
    main()