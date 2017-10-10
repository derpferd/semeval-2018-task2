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



