import yaml
import tweepy
import pandas as pd

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

def update_profile(name = 'MyName', url = 'http://mywebsite.com', location = 'MyCity', description = 'MyProfile'):
    api.update_profile(name, url, location, description)

def unretweet(tweet_id):
    try:
        api.unretweet(tweet_id)
    except tweepy.TweepError as e:
        print(e.reason)

def get_user_timeline(username):
    try:
        tweets = api.user_timeline(screen_name=username, count=200)
    except tweepy.TweepError as e:
        print(e.reason)
    else:
        return tweets

def get_trending_topics(woeid=1):  # 1 is the WOEID for worldwide
    try:
        trending = api.trends_place(id=woeid)
    except tweepy.TweepError as e:
        print(e.reason)
    else:
        return trending

def favorite_tweet(tweet_id):
    try:
        api.create_favorite(tweet_id)
    except tweepy.TweepError as e:
        print(e.reason)

def main():
    user = 'user_screen_name'
    tweet_id = 'tweet_id'

    # Update user profile
    update_profile()

    # Unretweet a tweet
    unretweet(tweet_id)

    # Favorite a tweet
    favorite_tweet(tweet_id)

    # Getting user timeline
    user_tweets = get_user_timeline(user)
    
    if user_tweets:
        # Create Tweets DataFrame
        data = pd.DataFrame(data=[tweet.text for tweet in user_tweets], columns=['Tweets'])

        # Print DataFrame
        print("\nData\n")
        print(data)

        # Print trending topics
        print("\nTrending Topics\n")
        trending = get_trending_topics()
        
        if trending:
            for value in trending:
                for trend in value['trends']:
                    print(trend['name'])

if __name__ == "__main__":
    main()