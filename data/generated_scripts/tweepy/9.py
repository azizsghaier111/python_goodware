import tweepy
import yaml

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

def unfollow_user(screen_name):
   try:
       api.destroy_friendship(screen_name)
       print(f"Unfollowed the user: {screen_name}")
   except tweepy.TweepError as e:
       print(e.reason)

def update_profile(name=None, url=None, location=None, description=None):
    try:
        api.update_profile(name, url, location, description)
        print("Profile updated")
    except tweepy.TweepError as e:
        print(e.reason)

def get_user_timeline(screen_name):
    tweets = api.user_timeline(screen_name=screen_name, count=200)
    for tweet in tweets:
        print(f"{tweet.user.name} - {tweet.text}")

def main():
    unfollow_user("user_to_unfollow")

    update_profile(name="New Name", url="http://newurl.com", 
                   location="New Location", description="New Description")

    print(f"\nUser timeline for user_to_get_timeline:")
    get_user_timeline("user_to_get_timeline")

if __name__ == "__main__":
    main()