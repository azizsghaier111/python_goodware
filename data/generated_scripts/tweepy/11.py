import yaml
import tweepy
import pandas as pd
import networkx as nx

with open("credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

CONSUMER_KEY = credentials['CONSUMER_KEY']
CONSUMER_SECRET = credentials['CONSUMER_SECRET']
ACCESS_TOKEN = credentials['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def unretweet(id):
   try:
       api.unretweet(id)
       print(f"Unretweeted the tweet: {id}")
   except tweepy.TweepError as e:
       print(e.reason)

def get_user_timeline(username):
    tweets = api.user_timeline(screen_name=username, count=200)
    for tweet in tweets:
        print(f"{tweet.user.name} - {tweet.text}")
    return tweets

def get_trending_topics(woeid=1):  # 1 is the WOEID for worldwide
    trending = api.trends_place(id = woeid)
    for value in trending:
        for trend in value['trends']:
            print(trend['name'])

def main():
    # unretweets
    unretweet(1234567890)  # replace with valid tweet ID

    # fetches timeline
    user_tweets = get_user_timeline("user_to_get_timeline")
    
    # pandas DataFrame
    data = pd.DataFrame(data=[tweet.text for tweet in user_tweets], columns=['Tweets'])
    print("\nData\n")
    print(data)  # prints dataframe of tweets text
    print("\nTrending Topics\n")
    get_trending_topics()

if __name__ == "__main__":
    main()