import tweepy
import yaml

# Load your Twitter API credentials
with open("credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authentification
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creating API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

def get_trending_topics(woeid = 1):
    """Function to get trending topics"""
    trending = api.trends_place(woeid)
    for value in trending:
        for trend in value['trends']:
            print(trend['name'])

def delete_tweet(tweet_id):
    """Function to delete a tweet"""
    api.destroy_status(tweet_id)

def get_following(user):
    """Function to list all following of a user"""
    for friend in tweepy.Cursor(api.friends, screen_name=user).items():
        print(friend.screen_name)

def main():
    # Define user and tweet_id here
    user = 'my_username'
    tweet_id = 1234567890

    print("\n*** Getting Trending Topics Worldwide ***")
    get_trending_topics()

    print("\n*** Deleting a Tweet ***")
    delete_tweet(tweet_id)

    print("\n*** Getting List of All Following ***")
    get_following(user)

if __name__ == "__main__":
    main()