""" Code Author: Jonathan Beaulieu"""
#Filename: run.py

#Purpose: Trains and Tests the data on the Models developed and prints out the result file for the respective input.
from __future__ import print_function
import os
import shutil
from scorer_semeval18 import main
from hopper import Tweet
from hopper.model_rand import RandModel
from hopper.model_naive_bayes_baselines import BernoulliNaiveBayesModel
from hopper.model_most_frequent_class import MostFrequentClassModel
from hopper.confusion_matrix import ConfusionMatrix
from hopper.scorer import Scorer

# The models to test
models = [RandModel,
          MostFrequentClassModel,
          BernoulliNaiveBayesModel]

train_test_ratio = 0.1  # Setting up Cross Validation value. This is the the number of folds over 1. Ex. if you want 10 folds then this should be 1/10.
langauges = {#"es": "Spanish", 
        "us": "English"}

if os.path.exists("raw_out"):
    shutil.rmtree("raw_out")
os.mkdir("raw_out")

data = "trial"  # This can be "train" or "trial"

data_root = os.path.join("data", data)


def count(data, label):  #Function to count the data available for the respective test/train class. Gives out the number of tweets available for each gold emoji in testing and training data
    return sum([1 for t in data if t.emoji == label])

#There are 2 parts of data that are being handled, .text file with the tweets listed out and .labels file which has a corresponding label(emoji) assigned to each and every tweet in the text file.
#The function returns the tweets as the output which a list of text and their corresponding emoji label.
	
def load_tweets(basepath): #Loads the twitter data file and respective labels file .
    text_path = basepath + ".text"
    labels_path = basepath + ".labels"

    # Read in input
    try:
        text_fp = open(text_path, 'r')
        text = text_fp.readlines()
        text_fp.close()
    except IOError:
        print("Had error reading from: {}".format(text_path))
    try:
        labels = open(labels_path, 'r').readlines()  # noqa
    except IOError:
        print("Had error reading from: {}".format(text_path))

    tweets = []  # type: List[Tweet]
    for text, label in zip(text, labels):
        tweets += [Tweet(text, int(label))]

    return tweets


for model_cls in models: # Repeats the loop for all the models present in models: BernoulliNaiveBayesModel,MostFrequentClassModel and RandModel
    print("============= {} =============".format(model_cls.__name__))  # Printing out the Model name that is being used
    for langauge in sorted(langauges, reverse=True):
        label_count = 20
        # Load tweets
        tweets = load_tweets(os.path.join(data_root, langauge + "_"+data))

        print("Doing {} cross folds".format(int(1 / train_test_ratio)), end="", flush=True)
        # Cross validation: In cross validation the data is divided into 10 parts , where 9 parts are used for training the model and one part is used for testing. 
        #This process is repeated so all the divided parts are used once as testing data.
        #For our project we are doing 10 fold cross validation.
        total_scorer = Scorer()

        for i in range(int(1 / train_test_ratio)):
            fold_scorer = Scorer()
            output_filename = os.path.join("raw_out", model_cls.__name__ + "." + str(i) + "." + langauge + "." + data + ".output.txt")
            gold_filename = os.path.join("raw_out", model_cls.__name__ + "." + str(i) + "." + langauge + "." + data + ".gold.txt")
            output_fp = open(output_filename, 'w') # Writing results to the output file
            gold_fp = open(gold_filename, 'w')

            model = model_cls()
            test_amount = int(len(tweets) * train_test_ratio) 
            test_data = tweets[i * test_amount: (i + 1) * test_amount]
            train_data = []
            if i > 0:
                train_data += tweets[0: i * test_amount]
            if i + 1 < int(1 / train_test_ratio):
                train_data += tweets[(i + 1) * test_amount:]

            model.train(train_data)

            matrix = ConfusionMatrix(label_count) # Generating the confusion matrix for the classifier, the value of label count is hardcoded to 20. Check confusion_matrix.py
            for tweet in test_data:
                output = model.predict(tweet.text)
                gold = tweet.emoji
                matrix.add(gold, output)
                fold_scorer.add(gold, output)
                total_scorer.add(gold, output)
                output_fp.write(str(output) + "\n")
                gold_fp.write(str(gold) + "\n")

            output_fp.close()
            gold_fp.close()

            print("------ {} Results ------".format(langauges[langauge]))
            #main(gold_filename, output_filename)
            print("{}".format(fold_scorer.get_score()))

            #Prints out the results
            print()
            print("----- Details -----")
            print("Training data len: {} Testing data len: {}".format(len(train_data), len(test_data)))
            print("Training data class counts: " + ", ".join([str(i) + ": " + str(count(train_data, i)) for i in range(label_count)]))
            print("Testing  data class counts: " + ", ".join([str(i) + ": " + str(count(test_data, i)) for i in range(label_count)]))
            print("--- Matrix ---\n" + str(matrix))
            print()
        print("Total score for all folds\n{}".format(total_scorer.get_score()))
        print()
    print()
