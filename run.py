import os
from model import Tweet
from model_rand import RandModel
import shutil


if os.path.exists("output"):
    shutil.rmtree("output")

os.mkdir("output")

langauges = ["es", "us"]

for langauge in langauges:
    # Read in input
    text = open(os.path.join("data", "trial", langauge+"_trial.text"), 'r').readlines()
    labels = open(os.path.join("data", "trial", langauge+"_trial.labels"), 'r').readlines()

    tweets = []
    for text, label in zip(text, labels):
        tweets += [Tweet(text, int(label))]

    model = RandModel()
    model.train(tweets[:-100])

    output_fp = open(os.path.join("output", langauge+".trial.output.txt"), 'w')
    gold_fp = open(os.path.join("output", langauge+".trial.gold.txt"), 'w')

    for tweet in tweets[-100:]:
        output = model.predict(tweet.text)
        gold = tweet.emoji
        output_fp.write(str(output)+"\n")
        gold_fp.write(str(gold)+"\n")

    output_fp.close()
    gold_fp.close()
