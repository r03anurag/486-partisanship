# 486 Partisanship Detection from Tweets
A model that can be used to predict the __political partisanship__ of a user based on one of their __tweets__.

---
## Description
Our model consists of the Bidirectional Encoder Representations from Transformers (BERT) architecture from HuggingFace. 
We obtained our data by collecting 9200 recent tweets from politicians of known political affiliations (20 Republicans and 20 Democrats) using the Twitter API, and split the data into 70% for training, 15% for validation and 15% for testing.
After training our model with the training set, we further fine-tuned the model's hyperparameters to prevent overfitting and underfitting using a grid search. Then, we evaluated our baseline and optimal models on the testing data and generated confusion matrices to visualize the models' performances. This repository consists of said models, along with the scripts used to obtain the data with the Twitter API, process the data, and train, validate, and evaluate the models. There are also folders of output CSVs containing the tweets obtained from the Twitter API, along with other files containing relevant data such as a file containing all the tweets, allTweets.csv, and a list of incorrect predictions made by the optimal model, incorrectPredictions.json.

---
## Getting Started
Take a look at the required dependencies and install them. After that, the order of running scripts is: _collectData.py_ __->__ uncomment line 70 in _processData.py_ if it is your first time running training after collecting data __->__ _trainAndEvaluate.py_ __->__ pick optimal hyperparameter combination for model based on highest accuracy on the validation set and obtain results __->__ _generatePlots.py_ -> plot confusion matrices for the baseline and optimal models.
### Dependencies

* Requests
* Pandas
* SciKit-Learn
* Torch
* SimpleTransformers
* Matplotlib

---
## Usage
Pull the data from Twitter using _collectData.py_. It takes around 3 hours to run due to rate limits.

Then, in _processData.py_, uncomment line 70 on your first run after running _collectData.py_ so that way the CSVs created by _collectData.py_ are aggregated into one CSV for easier processing.

After, run _trainAndEvaluate.py_ to prepare the data and find the training, validation, and testing sets, train the model, tune it, and evaluate it.

Lastly, read through the `lr_..._wd_....txt` files in the `results` folder and find the optimal hyperparameter combination by finding the result with the highest accuracy on the validation set, with ties being broken by the combination with the higher weight decay. Use the true positive (tp), true negative (tn), false positive (fp), and false negative (fn) values from these result files in _generatePlots.py_. Run _generatePlots.py_ to create confusion matrices that display the prediction summaries for the models.
### collectData.py
This file takes in no arguments. It gets the politician's user id, then pulls the necessary tweets' text from Twitter for each predetermined politician on Twitter. 20 are Republicans and 20 are Democrats.

There is a rate limit of 10 requests every 15 minutes (and 5 requests per user id every 15 minutes). The script __automatically__ respects and waits for the rate limit penalty to be over. Therefore, running this will take time, __about 3 hours__.

To run, simply run:
```
python3 collectData.py
```

The output will be in _output/democrats_ and _output/republicans_, where each has their respective CSVs of the politicians associated with that party. Each CSV corresponds to a politician and is named after them. Each CSV has 230 rows of tweets, where each row consists of these columns: Username (not id), Label (0 for Democrat; 1 for Republican), and Tweet with the text of the tweet. Each tweet is stripped of new lines at the end and new lines within the tweet are replaced with spaces. Quotations within the tweets are replaced with double `""` to escape them.
### processData.py 
#### Note that this program should not be run by you. It is run by _trainAndEvaluate.py_.
#### Running for the first time after running _collectData.py_? Make sure to uncomment line 72 in this file to ensure that you combine all CSVs created by _collectData.py_ into one for this program to use!
If not running this for the first time, keep line 70 commented out so that aggregating CSVs does not happen more than it needs to for efficiency.

Overall, this program can aggregate the CSVs created by collectData.py into one large CSV file called allTweets.csv, then load the data into a Pandas dataframe. 
Then, it will partition the data into training, validation, and testing data: 70% for training, 15% for validation and 15% for testing.
Running _trainAndEvaluate.py_ will call the load_and_partition_data function within this file to access the training, validation, and testing sets.

### trainAndEvaluate.py
To train, tune, and test the model, run this:
```
python3 trainAndEvaluate.py
```
This script will first run a grid search over the learning rate and weight decay to find the optimal combination that produces the highest accuracy on the validation set.
It will also store the performance metrics from evaluating on the testing set for these models so that we do not need to train again for the testing set, as training is very slow.
Lastly, it will save a list of the incorrect predictions made by the optimal model.

The output files of this will be the performance results for the different combinations of learning rates and weight decays, along with a text file called _incorrectPredictions.txt_ that contains the incorrect predictions that the model made for transparency.

Files are named `lr_<lr>_wd_<wd>.txt` where `<lr>` and `<wd>` are the respective learning rate and weight decay values. They are stored in the `results` folder.

__Each of these files contains the following information:__
* Learning rate (lr) and weight decay (wd) values.
* Validation performance.
* Test performance.

As for the file named `incorrectPredictions.txt`, it contains the incorrect predictions made by the model during testing.

### generatePlots.py
The baseline model's validation and testing results are stored in _lr_1e-05_wd_0.01.txt_. To find the optimal hyperparameter combination, go to the results folder and look through each file. 
Find the weights file with the highest validation accuracy and highest weight decay in case of ties, in our case _lr_0.0001_wd_0.0001.txt_. For the baseline and optimal model, use the true positive, true negative, false positive, false negative values from the results on the testing set for the tp, tn, fp, fn parameters in generatePlots.py's _generateConfusionMatrix_ function.

Then, to generate 2 confusion matrices to understand the performance of the model, run:
```
python3 generatePlots.py
```
The first confusion matrix will be for the baseline model while the second confusion matrix will be for the optimal model.

They will be saved to the following output files:
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