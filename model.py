from builtins import str as text


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
