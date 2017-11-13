# Author: Jonathan Beaulieu
from hopper import Model
from random import choice


class RandModel(Model):
    def train(self, tweets):
        self.classes = list(set([t.emoji for t in tweets]))

    def predict(self, text):
        return choice(self.classes)
