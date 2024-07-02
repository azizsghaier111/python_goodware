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

def update_user_profile(username, name=None, description=None, url=None, location=None):
    try:
        api.update_profile(username, name, url, location, description)
        print("Profile details updated successfully!")
    except tweepy.error.TweepError as e:
        print(e)

def get_retweets_of_tweet(tweet_id):
    retweets = api.retweets(tweet_id)
    for retweet in retweets:
        print(f"{retweet.user.name} retweeted: {retweet.text}")

def retweet(tweet_id):
    try:
        api.retweet(tweet_id)
        print("Retweeted successfully!")
    except tweepy.error.TweepError as e:
        print(e)

def get_trending_topics(location_id=1):
    trends = api.trends_place(location_id)
    for trend in trends[0]["trends"]:
        print(trend["name"])

def search_for_tweets(query, lang="en"):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang=lang).items(10)
    for tweet in tweets:
        print(f"{tweet.user.name}: {tweet.text}")

def main():
    print("\n1. UPDATING USER PROFILE DETAILS.")
    username = ""  # Fill it with a username
    name = ""  # Fill it with a name
    description = ""  # Fill it with a description
    url = ""  # Fill it with a url
    location = ""  # Fill it with a location
    update_user_profile(username, name, description, url, location)
    print("\n2. GETTING RETWEETS OF A TWEET.")
    tweet_id1 = ""  # Fill it with a tweet id
    get_retweets_of_tweet(tweet_id1)
    print("\n3. RETWEETING.")
    tweet_id2 = ""  # Fill it with a tweet id
    retweet(tweet_id2)
    print("\n4. GETTING TRENDING TOPICS.")
    get_trending_topics()
    print("\n5. SEARCHING FOR TWEETS.")
    search_query = "\"data science\" -filter:retweets"
    search_for_tweets(search_query)

if __name__ == "__main__":
    main()