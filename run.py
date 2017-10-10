""" Code Author: Jonathan Beaulieu"""

from __future__ import print_function
import os
import shutil
from scorer_semeval18 import main
from hopper import Tweet
from hopper.model_rand import RandModel
from hopper.model_naive_bayes_baselines import BernoulliNaiveBayesModel
from hopper.confusion_matrix import ConfusionMatrix
from hopper.model_most_frequent_class import MostFrequentClassModel
# The models to test
models = [RandModel,
          MostFrequentClassModel,BernoulliNaiveBayesModel]

if os.path.exists("raw_out"):
    shutil.rmtree("raw_out")
os.mkdir("raw_out")

langauges = {"es": "Spanish", "us": "English"}
trial_root = os.path.join("data", "train")

train_test_ratio = 0.1


def count(data, label):
    return sum([1 for t in data if t.emoji == label])


def load_tweets(basepath):
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


for model_cls in models:
    print("============= {} =============".format(model_cls.__name__))
    for langauge in sorted(langauges, reverse=True):
        label_count = 20
        # # Read in input
        # try:
        #     text_data_filename = os.path.join(trial_root, langauge + "_trial.text")
        #     text_fp = open(text_data_filename, 'r')
        #     text = text_fp.readlines()
        #     text_fp.close()
        # except IOError:
        #     print("Had error reading from: {}".format(text_data_filename))
        # labels = open(os.path.join("data", "trial", langauge + "_trial.labels"), 'r').readlines()  # noqa
        #
        # tweets = []  # type: List[Tweet]
        # for text, label in zip(text, labels):
        #     tweets += [Tweet(text, int(label))]

        tweets = load_tweets(os.path.join(trial_root, langauge + "_train"))

        print("Doing {} cross folds".format(int(1 / train_test_ratio)), end="", flush=True)
        for i in range(int(1 / train_test_ratio)):

            output_filename = os.path.join("raw_out", model_cls.__name__ + "." + str(i) + "." + langauge + ".trial.output.txt")
            gold_filename = os.path.join("raw_out", model_cls.__name__ + "." + str(i) + "." + langauge + ".trial.gold.txt")
            output_fp = open(output_filename, 'w')
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

            matrix = ConfusionMatrix(label_count)
            for tweet in test_data:
                output = model.predict(tweet.text)
                gold = tweet.emoji
                matrix.add(gold, output)
                output_fp.write(str(output) + "\n")
                gold_fp.write(str(gold) + "\n")

            output_fp.close()
            gold_fp.close()

            print("------ {} Results ------".format(langauges[langauge]))
            main(gold_filename, output_filename)
            print()
            print("----- Details -----")
            print("Training data len: {} Testing data len: {}".format(len(train_data), len(test_data)))
            print("Training data class counts: " + ", ".join([str(i) + ": " + str(count(train_data, i)) for i in range(label_count)]))
            print("Testing  data class counts: " + ", ".join([str(i) + ": " + str(count(test_data, i)) for i in range(label_count)]))
            print("--- Matrix ---\n" + str(matrix))
            print()
            # print(".", end="", flush=True)
        # print()
    print()
