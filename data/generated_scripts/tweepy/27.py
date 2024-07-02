import tweepy
import yaml
import pandas as pd

# Function to load Twitter API credentials from json file
def load_credentials(filename):
    with open(filename) as file:
        credentials = yaml.safe_load(file)
    return credentials

# Authenticate to Twitter
def authenticate(credentials):
    auth = tweepy.OAuthHandler(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
    auth.set_access_token(credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Get user profile details
def get_profile(api):
    return api.me()

# Post a tweet
def post_tweet(api, status):
    api.update_status(status)

# Unfavorite a tweet
def unfavorite_tweet(api, tweet_id):
    api.destroy_favorite(tweet_id)

# Function to get who I'm following
def get_following(api, screen_name):
    following = []
    for user in tweepy.Cursor(api.friends, screen_name=screen_name).items():
        following.append(user.screen_name)
    return following

# Function to get my followers
def get_followers(api, screen_name):
    followers = []
    for follower in tweepy.Cursor(api.followers, screen_name=screen_name).items():
        followers.append(follower.screen_name)
    return followers

# Search and return tweets
def search_tweets(api, text, lang='en', count=10):
    tweets = tweepy.Cursor(api.search_tweets, q=text, lang=lang, tweet_mode='extended').items(count)
    return [tweet.full_text for tweet in tweets]

def main():
    # Load the credentials from yaml file
    credentials = load_credentials('credentials.yaml')

    # Authenticate to Twitter API
    api = authenticate(credentials)

    # Get and print profile information
    profile = get_profile(api)
    print(f"Name: {profile.name}")
    print(f"Location: {profile.location}")

    # Post a tweet
    post_tweet(api, "Hello, Twitter!")

    # Unfavorite a tweet
    unfavorite_tweet(api, 'xxxx')

    # Get the lists of following and followers
    following = get_following(api, profile.screen_name)
    followers = get_followers(api, profile.screen_name)

    # Print the lists
    print(f"Following: {following}")
    print(f"Followers: {followers}")

    # Search for a keyword and print the tweets
    tweets = search_tweets(api, "Python programming", count=5)
    
    for tweet in tweets:
        print(tweet)

if __name__ == '__main__':
    main()