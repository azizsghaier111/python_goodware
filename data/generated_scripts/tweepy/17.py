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

def get_trending_topics(location_id=1):
    ''' Function to get the trending topics by given location id '''
    trending_topics = api.trends_place(location_id)
    topics = []

    for region in trending_topics:
        for trend in region['trends']:
            topics.append(trend['name'])

    return topics

def unfollow_user(username):
    ''' Function to unfollow a user by given username '''
    try:
        api.destroy_friendship(username)
        print(f"Successfully unfollowed {username}")
    except Exception as e:
        print(f"Error: {str(e)}")

def post_tweet(message):
    ''' Function to post a tweet '''
    try:
        api.update_status(message)
        print(f"Successfully posted tweet: {message}")
    except Exception as e:
        print(f"Error: {str(e})")

def main():
    # Get trending topics
    print("====== Trending Topics ======")
    trending_topics = get_trending_topics()
    print(trending_topics)
    pd.DataFrame(trending_topics, columns=['Trending Topic']).to_csv('trends.csv', index=False)

    # Unfollow a user 
    print("====== Unfollow a User ======")
    username = 'username_here'
    unfollow_user(username)

    # Post a tweet 
    print("====== Post a New Tweet ======")
    tweet_message = 'This is a test tweet'
    post_tweet(tweet_message)

if __name__ == "__main__":
    main()