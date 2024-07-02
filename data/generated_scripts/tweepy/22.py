import tweepy
import yaml

# Load credentials from yaml file
with open("twitter_credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def favorite_tweet(tweet_id):
    """Function to favorite a tweet."""
    try:
        api.create_favorite(tweet_id)
        print("Tweet favorited.")
    except tweepy.TweepError as e:
        print(e)

def retweet(tweet_id):
    """Function to retweet."""
    try:
        api.retweet(tweet_id)
        print("Tweet retweeted.")
    except tweepy.TweepError as e:
        print(e)

def update_profile(name, website, location, description):
    """Function to update the profile details."""
    try:
        api.update_profile(name, website, location, description)
        print("Profile updated successfully.")
    except tweepy.TweepError as e:
        print(e)

def update_profile_image(image_path):
    """Function to update the profile image."""
    try:
        api.update_profile_image(image_path)
        print("Profile image updated successfully.")
    except tweepy.TweepError as e:
        print(e)

def main():
    # Update profile
    name = "New Name"
    website = "http://www.newwebsite.com"
    location = "New Location"
    description = "New Description"
    update_profile(name, website, location, description)

    # Update profile image
    image_path = "path/to/new/image.jpg"
    update_profile_image(image_path)

    # Retweet a tweet
    tweet_id = "insert-tweet-id"
    retweet(tweet_id)

    # Favorite a tweet
    favorite_tweet(tweet_id)

if __name__ == "__main__":
    main()