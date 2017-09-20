from __future__ import print_function
import os
from hopper import RandModel, Tweet
import shutil
from typing import List


if os.path.exists("output"):
    shutil.rmtree("output")

os.mkdir("output")

langauges = ["es", "us"]
trial_root = os.path.join("data", "trial")

for langauge in langauges:
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

    model = RandModel()
    model.train(tweets[:-100])

    output_fp = open(os.path.join("output", langauge + ".trial.output.txt"), 'w')
    gold_fp = open(os.path.join("output", langauge + ".trial.gold.txt"), 'w')

    for tweet in tweets[-100:]:
        output = model.predict(tweet.text)
        gold = tweet.emoji
        output_fp.write(str(output) + "\n")
        gold_fp.write(str(gold) + "\n")

    output_fp.close()
    gold_fp.close()
