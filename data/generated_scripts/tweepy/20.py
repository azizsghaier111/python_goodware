def get_user_timeline(user_id=None, screen_name=None):
    """function to get the timeline of a specific user"""
    for status in tweepy.Cursor(api.user_timeline, id=user_id, screen_name=screen_name).items(10):
        print(status.text)
        
def follow_user(user_id=None, screen_name=None):
    """function to follow a specific user"""
    if user_id is not None:
        api.create_friendship(user_id)
    else:
        api.create_friendship(screen_name)

def get_followers_list(user_id=None, screen_name=None):
    """function to get the list of followers of a specific user"""
    for follower in tweepy.Cursor(api.followers, id=user_id, screen_name=screen_name).items(10):
        print(follower.screen_name)
    
# The main function
def main():
    """main function to call other functions"""
    print("*** Getting User's Timeline ***")
    get_user_timeline(screen_name="elonmusk")

    print("\n*** Following a user ***")
    follow_user(screen_name="BillGates")
    
    print("\n*** Getting User's followers ***")
    get_followers_list(screen_name="BarackObama")

    print("*** Getting Trending Topics Worldwide ***")
    get_trending_topics()

    print("\n*** Reading Tweets from Home Timeline ***")
    read_tweets()

    print("\n*** Sending Direct Message ***")
    user_id = "TwitterUserId"  #replace with valid twitter user id
    message = "This is a direct message"
    send_direct_message(user_id, message)

    print("\n*** Unfavoriting a Tweet ***")
    tweet_id = "12345"  # replace with valid tweet id
    unfav_tweet(tweet_id)
    
if __name__ == "__main__":
    main()