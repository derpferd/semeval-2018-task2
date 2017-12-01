# Author: Jonathan Beaulieu
from typing import List, Dict, Union, Iterator

ConfigValue = Union[float, str, int, bool]


class Tweet(object):
    def __init__(self, text: str, emoji: int) -> None:
        """
        Args:
            text(text): The text of the tweet.
            emoji(int): The id of the emoji for this tweet.
        """
        self.text = text
        self.emoji = emoji


class Model(object):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def get_extra_configs() -> List[Dict[str, ConfigValue]]:
        """
        Extra configs should in the form:
            {"name": "config_name", "default": 1.0}
        """
        return []

    def train(self, tweets: List[Tweet]) -> None:
        """
        Args:
            tweets(List[Tweet]): A list of tweets.
        """
        raise NotImplementedError("This is an abstract method")

    def predict(self, text: str) -> int:
        """
        Args:
            text(text): The text of the tweet to predict the emoji
        Return:
            int: The id of the emoji for the text.
        """
        raise NotImplementedError("This is an abstract method")

    def tokenize(self, text: str) -> List[str]:
        """
        Args:
            text(text): The text of the tweet
        Return:
            list: The list fo token which represent the text to the model
        """
        raise NotImplementedError("This is an abstract method")

    def batch_predict(self, texts: Iterator[str]) -> Iterator[int]:
        """
        Args:
            texts(Iterator[str]): A "list" of texts of the tweets to predict the emojis for.
        Return:
            Iterator[int]: The "list" of ids of the emojis for the texts.
        """
        return map(self.predict, texts)
