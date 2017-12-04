# Author: Jonathan Beaulieu

import json

from random import choice

from hopper import Model


class RandModel(Model):
    def train(self, tweets):
        self.classes = list(set([t.emoji for t in tweets]))

    def predict(self, text):
        return choice(self.classes)

    def save_model(self, path):
        json.dump(self.classes, open(path, "w"))

    def load_model(self, path):
        self.classes = json.load(open(path, "r"))
