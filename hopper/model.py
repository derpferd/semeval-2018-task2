# Author: Jonathan Beaulieu


class Tweet(object):
    def __init__(self, text, emoji):
        """
        Args:
            text(text): The text of the tweet.
            emoji(int): The id of the emoji for this tweet.
        """
        self.text = text
        self.emoji = emoji


class Model(object):
    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def get_extra_configs():
        """
        Extra configs should in the form:
            {"name": "config_name", "default": 1.0}
        """
        return []

    def train(self, tweets):
        """
        Args:
            tweets(List[Tweet]): A list of tweets.
        """
        raise NotImplementedError("This is an abstract method")

    def predict(self, text):
        """
        Args:
            text(text): The text of the tweet to predict the emoji
        Return:
            int: The id of the emoji for the text.
        """
        raise NotImplementedError("This is an abstract method")

    def batch_predict(self, texts):
        return map(self.predict, texts)
