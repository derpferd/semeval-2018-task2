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


class SVCModel(Model):
    """This model classifies tweets into any one of twenty classes
    using SVM classification.
    """
    def __init__(self):
        kernel = self.__class__.KERNEL
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                        reduce_len=True,
                                        strip_handles=True).tokenize

        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('{}svc'.format(kernel), SVC(kernel=kernel))])
        self.classif = SklearnClassifier(pipeline)

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "strip_handles", "default": True}]
        return super(SVCModel, SVCModel).get_extra_configs() + configs

    def train(self, tweets):
        def tweet_to_tuple(x):
            return (FreqDist(self.tokenizer(x.text)), x.emoji)

        # Generates tuples of all the tweets to form the corpus
        corpus = map(tweet_to_tuple, tweets)

        self.classif.train(corpus)

    def predict(self, text):
        return self.classif.classify(FreqDist(self.tokenizer(text)))


class LinearSVCModel(SVCModel):
    KERNEL = "linear"


class RBFSVCModel(SVCModel):
    KERNEL = "rbf"
