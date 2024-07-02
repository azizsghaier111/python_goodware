import tweepy
import yaml

# Load Credentials
with open("credentials.yaml", 'r') as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

consumer_key = credentials['CONSUMER_KEY']
consumer_secret = credentials['CONSUMER_SECRET']
access_token = credentials['ACCESS_TOKEN']
access_token_secret = credentials['ACCESS_TOKEN_SECRET']

# Authenticate
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Get UserDetails Function
def get_user_details(username):
    user = api.get_user(username)
    print("Details of this user are:")
    print("Username: ", user.screen_name)
    print("Name: ", user.name)
    print("Location: ", user.location)

# SearchTweets Function
def search_tweets(keyword, num_tweets):
    for tweet in tweepy.Cursor(api.search, q=keyword, lang="en").items(num_tweets):
        print(tweet.created_at, tweet.text)

# Get User Timeline Function
def get_user_timeline(username, num_tweets):
    tweets = api.user_timeline(screen_name=username, count=num_tweets)
    for tweet in tweets:
        print(tweet.text)

# Unfavorite a Tweet
def unfavorite_tweet(tweet_id):
    try:
        api.destroy_favorite(tweet_id)
        print("The tweet has been unfavorited successfully.")
    except Exception as e:
        print("Error:", e)

# Listen to Real-time Tweets
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

# ListenTweets Function
def listen_tweets(keyword):
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    myStream.filter(track=[keyword], is_async=True)

# Read Tweets HomeTimeline Function
def read_home_timeline(num_of_tweets):
    public_tweets = api.home_timeline(count=num_of_tweets)
    for tweet in public_tweets:
        print(tweet.text)

# Main Function
def main():
    # Getting user details
    username = "Enter the username here"
    get_user_details(username)

    # Searching for tweets
    keyword = "Python"
    num_tweets = 20
    search_tweets(keyword, num_tweets)

    # Getting a timeline of a particular user
    username = "Enter the username here"
    num_tweets = 20
    get_user_timeline(username, num_tweets)
    
    # Unfavorite a tweet
    tweet_id = "Enter your tweet_id here"
    unfavorite_tweet(tweet_id)
    
    # Listen to real-time tweets about Python
    listen_tweets(keyword)

    # Reading the most recent 10 tweets from your home timeline
    read_home_timeline(10)


if __name__ == "__main__":
    main()