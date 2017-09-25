from __future__ import print_function
import os
import shutil
from scorer_semeval18 import main
from typing import List
from hopper import Tweet
from hopper.model_rand import RandModel
from hopper.model_naive_bayes_baselines import MultinomialNaiveBayesModel, BernoulliNaiveBayesModel

# The models to test
models = [RandModel, MultinomialNaiveBayesModel, BernoulliNaiveBayesModel]

if os.path.exists("output"):
    shutil.rmtree("output")

os.mkdir("output")

langauges = {"es": "Spanish", "us": "English"}
trial_root = os.path.join("data", "trial")


for model_cls in models:
    print("============= {} =============".format(model_cls.__name__))
    for langauge in sorted(langauges, reverse=True):
        # Read in input
        try:
            text_data_filename = os.path.join(trial_root, langauge + "_trial.text")
            text_fp = open(text_data_filename, 'r')
            text = text_fp.readlines()
            text_fp.close()
        except IOError:
            print("Had error reading from: {}".format(text_data_filename))
        labels = open(os.path.join("data", "trial", langauge + "_trial.labels"), 'r').readlines()  # noqa

        tweets = []  # type: List[Tweet]
        for text, label in zip(text, labels):
            tweets += [Tweet(text, int(label))]

        model = model_cls()
        model.train(tweets[:-100])

        output_filename = os.path.join("output", model_cls.__name__ + "." + langauge + ".trial.output.txt")
        gold_filename = os.path.join("output", model_cls.__name__ + "." + langauge + ".trial.gold.txt")
        output_fp = open(output_filename, 'w')
        gold_fp = open(gold_filename, 'w')

        for tweet in tweets[-100:]:
            output = model.predict(tweet.text)
            gold = tweet.emoji
            output_fp.write(str(output) + "\n")
            gold_fp.write(str(gold) + "\n")

        output_fp.close()
        gold_fp.close()

        print("------ {} Results ------".format(langauges[langauge]))
        main(gold_filename, output_filename)
        print()
    print()
