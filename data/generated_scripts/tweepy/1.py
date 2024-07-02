import tweepy
import networkx as nx
import yaml
import pandas as pd

with open("credentials.yaml", 'r') as stream:
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

def get_trending_topics(woeid = 1):
    trending = api.trends_place(woeid)
    for value in trending:
        for trend in value['trends']:
            print(trend['name'])

def get_user_timeline(screen_name):
    tweets = api.user_timeline(screen_name=screen_name, count=200, tweet_mode="extended")
    for tweet in tweets:
        print(f"{tweet.user.name} said: {tweet.full_text}")

def favorite_tweet(tweet_id):
    api.create_favorite(tweet_id)
    print("Favorited the tweet")

def main():
    print("Trending Topics Worldwide:")
    get_trending_topics()

    print("Getting timeline of realDonaldTrump:")
    get_user_timeline("realDonaldTrump")

    print("Using the tweet id of the first tweet in the timeline to make it favourite")
    timeline = api.user_timeline("realDonaldTrump", count=1)
    for tweet in timeline:
        favorite_tweet(tweet.id)

if __name__ == "__main__":
    main()