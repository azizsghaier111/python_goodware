import tweepy
import yaml


def load_credentials(file):
    """Loads credentials from a yaml files."""
    with open(file, "r") as stream:
        try:
            credentials = yaml.safe_load(stream)
            return credentials
        except yaml.YAMLError as exception:
            print(exception)


def authenticate(credentials):
    """Authenticates to Twitter API."""
    auth = tweepy.OAuthHandler(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
    auth.set_access_token(credentials['ACCESS_TOKEN'], credentials['ACCESS_TOKEN_SECRET'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


def update_profile(api, name="", url="", location="", description=""):
    """Updates user profile."""
    api.update_profile(name, url, location, description)


def get_following(api):
    """Retrieves all users the authenticated user is following."""
    following = []
    for friend in tweepy.Cursor(api.friends).items():
        following.append(friend.screen_name)
    return following


def get_followers(api):
    """Retrieves all followers of the authenticated user."""
    followers = []
    for follower in tweepy.Cursor(api.followers).items():
        followers.append(follower.screen_name)
    return followers


def main():
    credentials_file = "twitter_credentials.yaml"
    credentials = load_credentials(credentials_file)
    api = authenticate(credentials)

    # Update profile
    name = "Your Name"
    url = "http://yourwebsite.com"
    location = "Your Location"
    description = "Your Description"
    update_profile(api, name, url, location, description)

    # Get list of all following
    following = get_following(api)
    print("You are following: ")
    for user in following:
        print(user)
    
    # Get list of all followers
    followers = get_followers(api)
    print("Your followers are: ")
    for user in followers:
        print(user)


if __name__ == "__main__":
    main()