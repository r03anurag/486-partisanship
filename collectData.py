
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
import json
# import tweepy
from requests_oauthlib import OAuth1


def get_tweets(user_id):
    user_id = "nielson_pa45889"
    tweet_count = 1000
    consumer_key = "I5jubR6rM8cOXluzo95cOokBF"
    consumer_secret = "qMwSoF2Euz79aFuFMH3j3J0NFNRWSIohhTUCKgv25hSBBbAKpK"
    access_token = "1727161020257296384-G9qaTe6xEovjAvQEsOz8nQwO3sZtV7"
    access_token_secret = "ji5EGD3N1HnksN8bJdqZdyOVRVKakZxMtQsTK6tDMG2Oh"
    
    url = f"https://api.twitter.com/2/users/%7B{user_id}%7D/tweets?max_results={tweet_count}"
    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request returned an error: {response.status_code}, {response.text}")

def write_tweets(tweets, party, user_id):
    """tweets is a json object with id, edit history, text keys.
    Party: 0 is Democrat; 1 is Republican"""
    with open("output.csv", "a") as output_file:
        for entry in tweets:
            text = entry["text"].strip()
            text_with_spaces = text.replace("\n", " ")
            escaped_quotes_text = text_with_spaces.replace('"', '""')
            final_text = '"' + escaped_quotes_text + '"'
            output_file.write(f'{user_id},{party},{final_text}\n')
def get_test_tweets():
    with open("test.json", 'r') as file:
        json_data = json.load(file)
        tweets = json_data["data"]
    return tweets
def write_data():
    democrat_usernames = ["POTUS", "BarackObama", "VP", "GovKathyHochul", "GovWhitmer", "GovernorShapiro", "GovTinaKotek", "SenatorMenendez", "SenatorBaldwin", "amyklobuchar", "SenCortezMasto", "maziehirono", "RepRaulGrijalva", "RepShriThanedar", "RepKweisiMfume", "RepLoriTrahan", "TulsiGabbard", "SecRaimondo", "DepSecTodman", "AmbassadorTai"]
    republican_usernames = ["realDonaldTrump", "Mike_Pence", "repkevinhern", "RepBradWenstrup", "RepMonicaDLC", "mtgreenee", "kayiveyforgov", "BobbyJindal", "MikeDeWine", "SarahHuckabee", "NikkiHaley", "votetimscott", "MarshaBlackburn", "SenTuberville", "JDVance1", "SenCapito", "RealBenCarson", "mikepompeo", "stevenmnuchin1", "SecBernhardt"]
    with open("output.csv", "w") as output_file:
        output_file.write("Username,Label,Tweet\n")
    tweets = get_test_tweets()  # test ver
    write_tweets(tweets, 0, "BarackObama")  # test ver
    # for user_id in democrat_usernames:
    #     # tweets = get_tweets(user_id)
    #     # write_tweets(tweets, 0, user_id)
    
    # for user_id in republican_usernames:
        # tweets = get_tweets(user_id)
        # write_tweets(tweets, 1, user_id)     

def main():
    write_data()

if __name__ == "__main__":
    main()
