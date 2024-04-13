from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Set up the WebDriver
driver_path = '/opt/homebrew/bin/chromedriver'
driver = webdriver.Chrome()

# Navigate to Twitter and log in

driver.get('https://twitter.com/login')

# # Enter your Twitter credentials
# username = driver.find_element(By.XPATH,'//input[@name="session[username_or_email]"]')
# password = driver.find_element(By.XPATH,'//input[@name="session[password]"]')
#
# username.send_keys('Derek.mg2@gmail.com')
# password.send_keys('zismAn-5fytje-wywqer')
# password.send_keys(Keys.RETURN)

# Set up the log in
time.sleep(3)
username = driver.find_element(By.XPATH,"//input[@name='text']")
username.send_keys("Derek_mg2")
next_button = driver.find_element(By.XPATH,"//span[contains(text(),'Next')]")
next_button.click()

time.sleep(3)
password = driver.find_element(By.XPATH,"//input[@name='password']")
password.send_keys("zismAn-5fytje-wywqer")
log_in = driver.find_element(By.XPATH,"//span[contains(text(),'Log in')]")
log_in.click()

# Navigate to user's profile
user_handle = "realDonaldTrump"
driver.get(f'https://twitter.com/{user_handle}')
time.sleep(3)  # Wait for the page to load

# Scroll down to load more tweets
last_height = driver.execute_script("return document.body.scrollHeight")

# while True:
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(2)  # Wait for more tweets to load
#
#     new_height = driver.execute_script("return document.body.scrollHeight")
#     if new_height == last_height:
#         break
#     last_height = new_height

all_tweets = set()

start_time = time.time()


# while len(all_tweets) < 2500:
while time.time() - start_time < 180:

    # Extract tweets
    tweets = driver.find_elements(By.XPATH,'//div[@data-testid="tweetText"]')

    for tweet in tweets:
        # print(tweet.text)
        all_tweets.add(tweet.text.strip())

    # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("window.scrollBy(0,1000)", "")

    time.sleep(1)  # Wait for more tweets to load

print(len(all_tweets))

# for tweet in all_tweets:
#     print(tweet)

# for tweet in tweets:
#     tweet_text = tweet.find_element_by_xpath('.//div[@lang="en"]').text
#     print(tweet_text)

# Clean up
driver.quit()
