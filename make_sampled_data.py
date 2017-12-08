#!/usr/bin/python3
""" Code Author: Jonathan Beaulieu
    This file takes the data in the folder `root_train` for the language
    `lang`, samples the data randomly such that each class has the same amount
    of tweets and saves it into the folder `root`.
"""
from random import shuffle, choice
import os
from shutil import rmtree

root_train = "./data/train"
root = "./sampled_data"
lang = "us"

# If the `root` directory already exists then delete it.
if os.path.exists(root):
    rmtree(root)
os.mkdir(root)

# Load labels and texts
labels = map(lambda x: x.strip(), open(os.path.join(root_train, lang+"_train.labels")).readlines())
texts = map(lambda x: x.strip(), open(os.path.join(root_train, lang+"_train.text")).readlines())

# Organize the text by class.
data = {}  # Key: class Value: List of tweet texts which belong under that class.
for label, text in zip(labels, texts):
    if label in data:
        data[label] += [text]
    else:
        data[label] = [text]

# Calculate the number of texts the smallest class has.
# This way we know we can take at most this many from each class.
min_count = min(map(len, data.values()))
print(min_count)
new_data = []

# Grab a random slice of texts from each class of the maximum amount we can.
for label in data:
    l_texts = data[label]
    shuffle(l_texts)
    new_data += zip(l_texts[:min_count], [label]*min_count)

# Mix it up!
shuffle(new_data)

# Prep the files.
l_fp = open(os.path.join(root, lang+"_sampled.labels"), "w")
t_fp = open(os.path.join(root, lang+"_sampled.text"), "w")

# Dump it out.
for t, l in new_data:
    l_fp.write(l+"\n")
    t_fp.write(t+"\n")

