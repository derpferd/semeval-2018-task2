# Author: Dennis Asamoah Owusu
import collections
from hopper import Model

#counts the occurence of each item passed to it
c = collections.Counter()


class MostFrequentClassModel(Model):
    def train(self, tweets):
        #extract all labels from tweets and feed to counter
        labels = [t.emoji for t in tweets]
        c.update(labels)
        for elem, count in c.most_common(1):
            self.most_frequent_class = elem

    def predict(self, text):
        return self.most_frequent_class
