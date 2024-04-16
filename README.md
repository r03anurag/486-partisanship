# 486 Partisanship Detection in Tweets
A model that can be used to predict the __political partisanship__ of a user based on one of their __tweets__.

---
## Description
Our model consists of the Bidirectional Encoder Representations from Transformers (BERT) architecture from HuggingFace. 
We obtained our data by collecting 9200 recent tweets from politicians of known political affiliations (20 Republicans and 20 Democrats) using the Twitter API, and split the data into 70% for training, 15% for validation and 15% for testing.
After training our model with the training set, we further fine tuned the model's hyperparameters to prevent overfitting and underfitting using a grid search. Then, we made a confusion matrix with a script based on our training and testing results. This repository consists of said model, along with the scripts used to obtain the data from the Twitter API, process the data, train the model, and evaluate the model. There are also folders of output CSVs containing the tweets obtained from the Twitter API, along with other files containing relevant data.

---
## Getting Started
Take a look at the required dependencies and install them. After that, the order of running scripts is: _collectData.py_ __->__ uncomment line 72 in _processData.py_ if it is your first time running training after collecting data __->__ _trainAndEvaluate.py_ __->__ pick optimal model based on highest weight decay value to get true positive/negatives and false positives/negatives __->__ _generatePlots.py_.
### Dependencies
```
requests~=2.31.0
pandas~=1.4.4
nltk~=3.8.1
scikit-learn~=1.2.1
torch~=2.2.2
simpletransformers~=0.70.0
matplotlib~=3.5.3
```

---
## Usage
Pull the data from Twitter using _collectData.py_. It takes around 3 hours to run due to rate limits.

Then, in _processData.py_ uncomment line 72 on your first run after running _collectData.py_ so that way the CSVs created by _collectData.py_ are aggregated into one CSV for efficiency.

After, run _trainAndEvaluate.py_ to prepare the data for, train the model, tune it, and evaluate it.

Lastly, find the optimal model based on which one has the highest weight decay value in the `results/lr..._...wd.txt` files and use their true positive (tp), true negative (tn), false positive (fp), false negative (fn) values in _generatePlots.py_. Run _generatePlots.py_ to create visuals representing the efficiency of the model.
### collectData.py
This file takes in no arguments. It gets the user's user id, then pulls the necessary tweets' text from Twitter for each predetermined user on Twitter. 20 are Republicans and 20 are Democrats.

There is a rate limit of 10 requests every 15 minutes (and 5 requests per user id every 15 minutes). The script __automatically__ respects and waits for the rate limit penalty to be over. Therefore, running this will take time, __about 3 hours__.

To run, simply run:
```
python collectData.py
```

The output will be in _output/democrats_ and _output/republicans_, where each has their respective CSVs of the politicians associated with that party. Each CSV corresponds to a politician and is named so. Each CSV has 230 rows of tweets, where each row consists of these columns: username (not id), partisanship (0 for Democrat; 1 for Republican), tweet text. Each tweet is stripped of new lines at the end and new lines in the tweet are replaced with spaces. Quotations in the tweets are replaced with double `""` to espace them.
### processData.py 
#### Note that this program should not be run by you. It is run by _trainAndEvaluate.py_.
#### Running for the first time after running _collectData.py_? Make sure to uncomment line 72 in this file to ensure that you combine all CSVs created by _collectData.py_ into one for this program to use!
If not running this for the first time, make sure to comment line 72 so that aggregating CSVs does not happen more than it needs to for efficiency.

Overall, this program can aggregate the CSVs created by collectData.py, then it will load the data into Panda's dataframes. Lastly, it will partition the data into test, validation, and training data: 70% for training, 15% for validation and 15% for testing.
The input file will be "allTweets.csv" by default.
The file "allTweets.csv" will be created after running _trainAndEvaluate.py_, which will call the functions within this program.

### trainAndEvaluate.py
To process the data, train, validate, and evaluate the model, run this:
```
python trainAndEvaluate.py
```

The output files of this will be the performance results for different combinations of learning rates and weight decays, along with a text file called _incorrectPredictions.txt_ that contains the incorrect predictions that the model made for transparency.

Files named `lr_<lr>_wd_<wd>.txt` where `<lr>` and `<wd>` are the respective learning rate and weight decay values.

__Each of these files contains the following information:__
* Learning rate (lr) and weight decay (wd) values.
* Validation performance.
* Test performance.

As for the file named `incorrectPredictions.txt`, it ontains the incorrect predictions made by the model during testing.

### generatePlots.py
Go to the results folder and look through each weights file. 
Find the weights file with the highest weight decay value. Use its true positive, true negative, false positive, false negative values for tp, tn, fp, fn parameters in generatePlots.py's _generateConfusionMatrix_ function.

Then, to generate 2 confusion matrixes to understand the performance of the model, run:
```
python generatePlots.py
```
The first confusion matrix will be for the optimal model while the second confusion matrix will be for the baseline model.

They will be in the following output files:
* `confusionMatrixBaseline.png`
* `confusionMatrixOptimal.png`

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