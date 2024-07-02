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

# Update profile
def update_profile(api, name=None, description=None, location=None):
    api.update_profile(name=name, description=description, location=location)
    
# Function to get who I'm following
def get_following(api, screen_name):
    following = []
    for user in tweepy.Cursor(api.friends, screen_name=screen_name).items():
        following.append(user.screen_name)
    return following

# Function to get my followers
def get_followers(api, screen_name):
    followers = []
    for follower in tweepy.Cursor(api.followers, screen_name=screen_name).items(10):
        followers.append(follower.screen_name)
    return followers

def search_tweets(api, keyword, lang, count):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang=lang, tweet_mode='extended').items(count):
        tweets.append(tweet.full_text)
    return tweets

def main():
    # Load credentials and authenticate
    credentials = load_credentials('credentials.yaml')
    api = authenticate(credentials)
    
    # Get and print profile information
    profile = get_profile(api)
    print('Profile:')
    print('Name: ' + profile.name)
    print('Location: ' + profile.location)
    print('Following: ' + str(profile.friends_count))
    print('Followers: ' + str(profile.followers_count))
    
    # Print the people I'm following
    print('People I am following:')
    following = get_following(api, profile.screen_name)
    print(following)
    
    # Print my followers
    print('My followers:')
    followers = get_followers(api, profile.screen_name)
    print(followers)
    
    # Search for a keyword and print the tweets
    keyword = 'python programming'
    print(f'Tweets related to "{keyword}":')
    tweets = search_tweets(api, keyword, 'en', 10)
    for tweet in tweets:
        print(tweet)
    
if __name__ == "__main__":
    main()