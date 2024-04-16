"""
collectData.py

Pulls 230 tweets from Twitter API for each predetermined politician,
then parses and formats the data into CSV.
"""
import requests
import json
import os
import time


def make_request(url, params):
    """
    This function makes a request with the given bearer token and handles the results accordingly.
    
    Args:
        url (string): the api endpoint.
        params (dictionary): the parameters to the api endpoint.
    
    Raises:
        Exception: api request returned an error.
    
    Returns:
        dictionary: api endpoint's json response converted to python dictionary.
    """
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAKo6tQEAAAAAo5vBqStnM8HhipqSiUeMM3T9Kto%3DRc4JpFfsn4V6rhSIzWKN26IMs13rpiLlsa9zPpAUAm2079Fo8p"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 429:
        reset_time = int(response.headers.get('x-rate-limit-reset'))
        wait_time = max(0, reset_time - int(time.time()))
        print(f"Rate limit exceeded. Waiting for {wait_time} seconds until reset.")
        time.sleep(wait_time)  # Wait until rate limit resets
        return make_request(url, params)  # Retry the request
    elif response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request returned an error: {response.status_code}, {response.text}")


def get_tweets(username):
    """
    This function gets the username's user id and fetches 230 tweets.

    Args:
        username (string): twitter user's actual username.

    Raises:
        Exception: Error response from username-user_id translation api endpoint.

    Returns:
        list (strings): tweets' text data
    """
    tweets = []
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAKo6tQEAAAAAo5vBqStnM8HhipqSiUeMM3T9Kto%3DRc4JpFfsn4V6rhSIzWKN26IMs13rpiLlsa9zPpAUAm2079Fo8p"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    
    # translate username to user id. will never go over this api rate limit
    # as long as we respect the tweets rate limit since this one is much more lenient
    user_endpoint = f"https://api.twitter.com/2/users/by/username/{username}"
    response = requests.get(user_endpoint, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request returned an error: {response.status_code}, {response.text}")
    data = response.json()
    user_id = data["data"]["id"]
    
    # Twitter API v2 endpoint for fetching user tweets
    endpoint = f"https://api.twitter.com/2/users/{user_id}/tweets"
    
    # 230 tweets per user so that can have 9200 tweets total
    params = {
        "max_results": 100,  # max per page is 100        
    }
    count = 0
    
    # get 200 tweets
    while count < 2:
        data = make_request(endpoint, params) 
        count += 1
        if data.get("errors"):
            print(data.get("errors"))
        next_token = data["meta"].get("next_token")
        params["pagination_token"] = next_token
        tweets.extend(data.get("data", []))
        if not next_token:
            return tweets  # stop if no more to find
    # get last 30 tweets
    if next_token:
        params["max_results"] = 30
        response = requests.get(endpoint, headers=headers, params=params)
        data = make_request(endpoint, params)
        if data.get("errors"):
            print(data.get("errors"))
        tweets.extend(data.get("data", []))
    
    return tweets


def write_tweets(tweets, party, username):
    """
    This function writes the tweets for a user in their corresponding CSV file. No return value.

    Args:
        tweets (list of strings): list of this user's 230 most recent tweets as strings.
        party (int): 0 is Democrat; 1 is Republican.
        username (string): The user's actual Twitter username.
    """

    party_name = "republicans"
    if party == 0:
        party_name = "democrats"
    
    with open(f"output/{party_name}/{username}_output.csv", "w", encoding="utf-8") as output_file:
        output_file.write("Username,Label,Tweet\n")
        for entry in tweets:
            # process the tweet text data: replace new lines with spaces,
            # replace quotations with double quotes, and add quotations around the text
            text = entry["text"].strip()
            text_with_spaces = text.replace("\n", " ")
            escaped_quotes_text = text_with_spaces.replace('"', '""')
            final_text = '"' + escaped_quotes_text + '"'
            output_file.write(f'{username},{party},{final_text}\n')


def get_test_tweets():
    """
    This function was used to test the functionality of other functions originally.

    Returns:
        list of strings: tweets' text in list
    """
    with open("test.json", 'r', encoding="utf-8") as file:
        json_data = json.load(file)
        tweets = json_data["data"]
    return tweets


def write_data():
    """
    For 20 predetermined Democrat and 20 predetermined Republican politicans,
    get their tweets and write them to CSV files.
    """
    democrat_usernames = [
        "POTUS", "BarackObama", "VP", "GovKathyHochul", 
        "DepSecTodman", "AmbassadorTai", "RepLoriTrahan",
        "TulsiGabbard", "SecRaimondo", "GovWhitmer",
        "GovernorShapiro", "GovTinaKotek", "SenatorMenendez",
        "SenatorBaldwin", "amyklobuchar", "SenCortezMasto",
        "maziehirono", "RepRaulGrijalva", "RepShriThanedar",
        "RepKweisiMfume", 
    ]
    republican_usernames = [
        "realDonaldTrump", "Mike_Pence", "stevenmnuchin1", 
        "SecBernhardt", "repkevinhern", "RepBradWenstrup",
        "RepMonicaDLC", "mtgreenee", "kayiveyforgov", "BobbyJindal",
        "MikeDeWine", "SarahHuckabee", "NikkiHaley", "votetimscott",
        "MarshaBlackburn", "SenTuberville", "JDVance1", "SenCapito",
        "RealBenCarson", "mikepompeo"
    ]

    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    rep = "output/republicans"
    if not os.path.exists(rep):
        os.makedirs(rep)
    dem = "output/democrats"
    if not os.path.exists(dem):
        os.makedirs(dem)
    
    for user_id in republican_usernames:
        tweets = get_tweets(user_id)
        write_tweets(tweets, 1, user_id)    
    for user_id in democrat_usernames:
        tweets = get_tweets(user_id)
        write_tweets(tweets, 0, user_id) 


def main():
    """Main function that calls the logic of the function.
    """
    write_data()


if __name__ == "__main__":
    main()
