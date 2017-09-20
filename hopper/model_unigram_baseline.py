from hopper import Model
from random import randint


class BaselineModel(Model):
    def train(self, tweets):
        pass

    def predict(self, text):
        return randint(0, 19)
