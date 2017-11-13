""" Code Author: Jonathan Beaulieu
"""
from __future__ import division
from hopper import Model
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class LinearSVCModel(Model):
    """This model classifies tweets into any one of twenty classes
    using SVM classification.
    """
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                        reduce_len=True,
                                        strip_handles=True).tokenize

        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('linearsvc', SVC(kernel="linear"))])
        self.classif = SklearnClassifier(pipeline)

    def train(self, tweets):
        def tweet_to_tuple(x):
            return (FreqDist(self.tokenizer(x.text)), x.emoji)

        # Generates tuples of all the tweets to form the corpus
        corpus = map(tweet_to_tuple, tweets)

        self.classif.train(corpus)

    def predict(self, text):
        return self.classif.classify(FreqDist(self.tokenizer(text)))
