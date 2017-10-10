import collections
from hopper import Model
from random import randint

c = collections.Counter()

class MostFrequentClassModel(Model):
    def train(self, tweets):
        labels = [t.emoji for t in tweets]
        c.update(labels)
        for elem, count in c.most_common(1):
            self.most_frequent_class = elem

    def predict(self, text):
        return self.most_frequent_class



'''
c = collections.Counter()

with open('/home/dennis/coding/semeval-2018-task2/data/trial/us_trial.labels') as f:
    content = f.readlines()

content = [x.strip() for x in content]
c.update(content)
for letter, count in c.most_common(1):
    key = letter

print (key)
'''
