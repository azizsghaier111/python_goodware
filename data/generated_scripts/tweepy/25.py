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
api = tweepy.API(auth)

def unretweet(api, tweet_id):
    try:
        api.unretweet(tweet_id)
        print(f'Tweet with id {tweet_id} was successfully unretweeted.')
    except tweepy.TweepError as e:
        print(e.reason)

def user_details(api, username):
    try:
        user = api.get_user(screen_name=username)
        print(f'Username: {user.screen_name}')
        print(f'Followers: {user.followers_count}')
        print(f'Following: {user.friends_count}')
        print(f'Number of tweets: {user.statuses_count}')
    except tweepy.TweepError as e:
        print(e.reason)
        
def dm_user(api, user_id, message):
    try:
        direct_message = api.send_direct_message(user_id, message)
        print(f'Direct message has been sent to {user_id} successfully.')
    except tweepy.TweepError as e:
        print(e.reason)

def main():
    unretweet(api, 'tweet_id')
    user_details(api, 'username')
    dm_user(api, 'user_id', 'message')
    
if __name__ == "__main__":
    main()