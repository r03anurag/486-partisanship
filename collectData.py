
# # This Python Script will set up authentication with X API

# import requests
#
# def get_tweets(user_id, tweet_count, bearer_token):
#     tweets = []
#     # Twitter API v2 endpoint for fetching user tweets
#     #endpoint = f"https://api.twitter.com/2/users/{user_id}/tweets"
#     endpoint = 'https://api.twitter.com/2/users/me'
#     headers = {"Authorization": f"Bearer {bearer_token}"}
#     params = {
#         "max_results": 10,  # Max results per request (API's limit)
#         "tweet.fields": "created_at",  # Additional tweet fields you want to include
#     }
#
#     while len(tweets) < tweet_count:
#         response = requests.get(endpoint, headers=headers, params=params)
#         if response.status_code == 200:
#             print("YESSSS")
#             return None
#         if response.status_code != 200:
#             raise Exception(f"Request returned an error: {response.status_code}, {response.text}")
#
#         data = response.json()
#         tweets.extend(data.get("data", []))
#
#         # Check if there's a next_token to paginate
#         next_token = data.get("meta", {}).get("next_token", None)
#         if not next_token or len(tweets) >= tweet_count:
#             break  # Exit loop if there's no more data to fetch or we've reached the desired count
#
#         params["pagination_token"] = next_token  # Set pagination token for next request
#
#     return tweets[:tweet_count]  # Return the requested number of tweets
#
# # User ID for the account you're interested in and the number of tweets to fetch
# user_id = "realDonaldTrump"
# tweet_count = 1
#
# # Your Bearer Token from the Twitter Developer Portal
# bearer_token = "AAAAAAAAAAAAAAAAAAAAADcitAEAAAAAQuUWW6ojfg5F3vzZVIQKRADDEe8%3DRhZZ7ZTmvfgvBRwRmATbVpBBkiB2lhcHYVVATCv1yvDvO9ZKVd"  # Or directly insert your token as a string
#
# # Fetch the tweets
# tweets = get_tweets(user_id, tweet_count, bearer_token)
#
# print(f"Fetched {len(tweets)} tweets.")


import requests
import tweepy
from requests_oauthlib import OAuth1


def get_tweets(user_id, tweet_count, consumer_key, consumer_secret, access_token, access_token_secret):
    url = f"https://api.twitter.com/2/users/%7B{user_id}%7D/tweets?max_results={tweet_count}"
    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request returned an error: {response.status_code}, {response.text}")


def write_data(filename):
    user_id = "nielson_pa45889"
    tweet_count = 10
    consumer_key = "I5jubR6rM8cOXluzo95cOokBF"
    consumer_secret = "qMwSoF2Euz79aFuFMH3j3J0NFNRWSIohhTUCKgv25hSBBbAKpK"
    access_token = "1727161020257296384-G9qaTe6xEovjAvQEsOz8nQwO3sZtV7"
    access_token_secret = "ji5EGD3N1HnksN8bJdqZdyOVRVKakZxMtQsTK6tDMG2Oh"

    tweets = get_tweets(user_id, tweet_count, consumer_key, consumer_secret, access_token, access_token_secret)
    print(tweets)

    democrat_usernames = ["POTUS", "BarackObama", "VP", "GovKathyHochul", "GovWhitmer", "GovernorShapiro", "GovTinaKotek", "SenatorMenendez", "SenatorBaldwin", "amyklobuchar", "SenCortezMasto", "maziehirono", "RepRaulGrijalva", "RepShriThanedar", "RepKweisiMfume", "RepLoriTrahan", "TulsiGabbard", "SecRaimondo", "DepSecTodman", "AmbassadorTai"]
    republican_usernames = ["realDonaldTrump", "Mike_Pence", "repkevinhern", "RepBradWenstrup", "RepMonicaDLC", "mtgreenee", "kayiveyforgov", "BobbyJindal", "MikeDeWine", "SarahHuckabee", "NikkiHaley", "votetimscott", "MarshaBlackburn", "SenTuberville", "JDVance1", "SenCapito", "RealBenCarson", "mikepompeo", "stevenmnuchin1", "SecBernhardt"]
    all_usernames = democrat_usernames + republican_usernames

    num_democrats = len(democrat_usernames)
    num_tweets_per_user = 10000 / len(all_usernames)

    with open(filename, 'a', encoding="utf8") as output_file:
        output_file.write("Username,Label,Tweet\n")

        for i, user_id in enumerate(all_usernames):

            # Democrats are 0 and Republicans are 1
            label = 0 if i < num_democrats else 1

            tweets_written = 0

            while tweets_written < num_tweets_per_user:

                ###########################################################
                # Guidelines for writing to csv file:
                # Do not put any spaces before or after commas
                # The tweets in the tweets column must be enclosed in double quotes
                # TODO: Remove all newline characters within tweet
                ###########################################################

                # tweets = get_tweets(user_id)
                #   for tweet in tweets:
                output_file.write(f"{user_id},{label},\"{tweet}\"\n")
                tweets_written += 1


def main():
    write_data("tweets.csv")


if __name__ == "__main__":
    main()
