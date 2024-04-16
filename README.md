# 486 Partisanship Detection in Tweets
A model that can be used to predict the __political partisanship__ of a user based on one of their __tweets__.

---
## Description
Our model consists of the Bidirectional Encoder Representations from Transformers (BERT) architecture from HuggingFace. It was trained from 230 recent tweets from 40 different politicians (20 Republicans and 20 Democrats), with a 80% training 20% testing data split, all to create word embeddings of the tweets. After, we further fine tuned the model parameters to prevent overfitting and underfitting. This repository consists of said model, along with the scripts used to obtain the data from the Twitter API, process the data, train the data, and evaluate the data. There are also folders of output CSVs containing the tweets obtained from the Twitter API, along with other files containing relevant data.

---
## Getting Started
Take a look at the required dependencies and install them. After that, the order of running scripts is: _collectData.py_ __->__ _processData.py_ __->__ _trainAndEvaluate.py_
### Dependencies
```
pytorch
etc.
```
---
## Usage
Pull the data from Twitter using _processData.py_. It takes around 3 hours to run due to rate limits.

After, run _processData.py_ to prepare the data for training and testing usage.

Lastly, run _trainAndEvaluate.py_ to train the model, tune it, and evaluate it.
### collectData.py
This file takes in no arguments. It gets the user's user id, then pulls the necessary tweets' text from Twitter for each predetermined user on Twitter. 20 are Republicans and 20 are Democrats.

There is a rate limit of 10 requests every 15 minutes (and 5 requests per user id every 15 minutes). The script __automatically__ respects and waits for the rate limit penalty to be over. Therefore, running this will take time, __about 3 hours__.

To run, simply run:
```
python collectData.py
```

The output will be in _output/democrats_ and _output/republicans_, where each has their respective CSVs of the politicians associated with that party. Each CSV corresponds to a politician and is named so. Each CSV has 230 rows of tweets, where each row consists of these columns: username (not id), partisanship (0 for Democrat; 1 for Republican), tweet text. Each tweet is stripped of new lines at the end and new lines in the tweet are replaced with spaces. Quotations in the tweets are replaced with double `""` to espace them.
### processData.py
```
python processData.py
```
### trainAndEvaluate.py
```
python trainAndEvaluate.py
```
---
## Authors
* Derek Goh
* Jooho Lee
* James Leung
* Anirudh Ramprasad
* Anurag Renduchintala 
### Contibuting
Please email us at [jameleu@umich.edu](mailto:jameleu@umich.edu) if you would like to contribute.

---
## License
N/A