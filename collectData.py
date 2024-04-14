import requests
import json
import os
import time


    
def make_request(url, params):
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
    # endpoint = 'https://api.twitter.com/2/users/me'  # free test point
    
    # 230 tweets per user so that can have 9200 tweets total
    params = {
        "max_results": 100,  # max per page is 100        
    }
    count = 0
    
    # test one api call
    # response = requests.get(endpoint, headers=headers)
    # data = response.json()
    # if data.get("errors"):
    #     print(data.get("errors"))
    # print("response:", data)
    # next_token = data["meta"].get("next_token")
    # print("token:", next_token)
    # tweets.extend(data.get("data", []))
    # return tweets

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
    """tweets is a json object with id, edit history, text keys.
    Party: 0 is Democrat; 1 is Republican"""

    party_name = "republicans"
    if party == 0:
        party_name = "democrats"
    
    with open(f"output/{party_name}/{username}_output.csv", "w") as output_file:
        output_file.write("Username,Label,Tweet\n")
        for entry in tweets:
            text = entry["text"].strip()
            text_with_spaces = text.replace("\n", " ")
            escaped_quotes_text = text_with_spaces.replace('"', '""')
            final_text = '"' + escaped_quotes_text + '"'
            output_file.write(f'{username},{party},{final_text}\n')
def get_test_tweets():
    with open("test.json", 'r') as file:
        json_data = json.load(file)
        tweets = json_data["data"]
    return tweets
def write_data():
    # old_dems = ["POTUS", "BarackObama", "VP", "GovKathyHochul", "GovWhitmer", "GovernorShapiro", "GovTinaKotek", "SenatorMenendez", "SenatorBaldwin", "amyklobuchar", "SenCortezMasto", "maziehirono", "RepRaulGrijalva", "RepShriThanedar", "RepKweisiMfume", ]
    # optional_dems = ["DepSecTodman", "AmbassadorTai"]
    # optional_reps = ["stevenmnuchin1", "SecBernhardt"]
    democrat_usernames = ["RepLoriTrahan", "TulsiGabbard", "SecRaimondo"]
    republican_usernames = ["realDonaldTrump", "Mike_Pence", "repkevinhern", "RepBradWenstrup", "RepMonicaDLC", "mtgreenee", "kayiveyforgov", "BobbyJindal", "MikeDeWine", "SarahHuckabee", "NikkiHaley", "votetimscott", "MarshaBlackburn", "SenTuberville", "JDVance1", "SenCapito", "RealBenCarson", "mikepompeo"]

    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    rep = "output/republicans"
    if not os.path.exists(rep):
        os.makedirs(rep)
    dem = "output/democrats"
    if not os.path.exists(dem):
        os.makedirs(dem)
    # tweets = get_test_tweets()  # test ver
    # write_tweets(tweets, 0, "BarackObama")  # test ver
    # write_tweets(tweets, 1, "realDonaldTrump")  # test ver
    
    # tweets = get_tweets("BarackObama")  # real api single call test ver
    # write_tweets(tweets, 0, "BarackObama")  # real api single call test ver
    

    
    for user_id in republican_usernames:
        tweets = get_tweets(user_id)
        write_tweets(tweets, 1, user_id)    
    for user_id in democrat_usernames:
        tweets = get_tweets(user_id)
        write_tweets(tweets, 0, user_id) 

def main():
    write_data()

if __name__ == "__main__":
    main()
