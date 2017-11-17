#!/usr/bin/python3
from random import shuffle, choice
import os
from shutil import rmtree

root_train = "/home/csgrads/beau0307/nlp/semeval-2018-task2/data/train"
root = "./sampled_data"
lang = "us"

if os.path.exists(root):
    rmtree(root)
os.mkdir(root)

labels = map(lambda x: x.strip(), open(os.path.join(root_train, lang+"_train.labels")).readlines())
texts = map(lambda x: x.strip(), open(os.path.join(root_train, lang+"_train.text")).readlines())

data = {}
for label, text in zip(labels, texts):
    if label in data:
        data[label] += [text]
    else:
        data[label] = [text]

min_count = min(map(len, data.values()))
print(min_count)
#new_labels = []
#new_text = []
new_data = []

for label in data:
    #new_labels += [label]*min_count
    l_texts = data[label]
    shuffle(l_texts)
    new_data += zip(l_texts[:min_count], [label]*min_count)

shuffle(new_data)

l_fp = open(os.path.join(root, lang+"_sampled.labels"), "w")
t_fp = open(os.path.join(root, lang+"_sampled.text"), "w")

for t, l in new_data:
    l_fp.write(l+"\n")
    t_fp.write(t+"\n")

