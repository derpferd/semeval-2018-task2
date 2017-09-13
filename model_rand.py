from model import Model
from random import randint


class RandModel(Model):
    def train(self, tweets):
        pass

    def predict(self, text):
        return randint(0, 19)
