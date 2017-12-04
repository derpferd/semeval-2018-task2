""" Author: Jonathan Beaulieu
"""
from __future__ import division

from typing import List, Optional

from nltk.classify import SklearnClassifier
from nltk.probability import FreqDist
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from hopper import Model, Tweet


class SVCModel(Model):
    """This model classifies tweets into any one of twenty classes
    using SVM classification.
    """

    def __init__(self, kernel: str = "") -> None:
        # Setup tweet tokenizer note this is the same as in our baseline. For a full description checkout the
        # model_naive_bayes_baselines source file.
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                        reduce_len=True,
                                        strip_handles=True).tokenize

        # Here we create the pipeline for the classifier.
        # The TfidfTransformer is the same as in our baseline. For a full description checkout the
        # model_naive_bayes_baselines source file.
        # The SVC sets up a Support Vector Machine classifier with the configured kernel.
        # In this case it is either a linear or a radial basis function kernel.
        # The details for the above items are discussed in the model's readme.
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('{}svc'.format(kernel), SVC(kernel=kernel))])
        self.classif = SklearnClassifier(pipeline)

    def train(self, tweets: List[Tweet]) -> None:
        def tweet_to_tuple(x):
            return (FreqDist(self.tokenizer(x.text)), x.emoji)

        # Generates tuples of all the tweets to form the corpus
        corpus = map(tweet_to_tuple, tweets)

        # Train this model!
        self.classif.train(corpus)

    def predict(self, text):
        return self.classif.classify(FreqDist(self.tokenizer(text)))


class LinearSVCModel(SVCModel):
    def __init__(self):
        super().__init__("linear")


class RBFSVCModel(SVCModel):
    def __init__(self):
        super().__init__("rbf")


class LinearSVC2Model(Model):
    """This model classifies tweets into any one of twenty classes
    using SVM classification.
    """

    def __init__(self, balanced=False, C=1.0, dual=True, tol=1e-4, max_iter=1000, loss="squared_hinge") -> None:
        # Setup tweet tokenizer note this is the same as in our baseline. For a full description checkout the
        # model_naive_bayes_baselines source file.
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                        reduce_len=True,
                                        strip_handles=True).tokenize

        # set class_weight to None unless the 'balanced' has been set to true in the config
        class_weight = None  # type: Optional[str]
        if balanced:
            class_weight = "balanced"

        # Here we create the pipeline for the classifier.
        # The TfidfTransformer is the same as in our baseline. For a full description checkout the
        # model_naive_bayes_baselines source file.
        # The LinearSVC sets up a Linear Support Vector Machine classifier. This is different because than using SCV
        # with a Linear kernel because it uses liblinear as a backend instead of libsvm. This makes it run a lot faster.
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('linearsvc', LinearSVC(class_weight=class_weight, C=C,
                                                     dual=dual, tol=tol, max_iter=max_iter,
                                                     loss=loss))])
        self.classif = SklearnClassifier(pipeline)

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "balanced", "default": False},
                   {"name": "C", "default": 1.0},
                   {"name": "dual", "default": True},
                   {"name": "tol", "default": 1e-4},
                   {"name": "max_iter", "default": 1000},
                   {"name": "loss", "default": "squared_hinge"}]  # add config for balanced.
        return super(LinearSVC2Model, LinearSVC2Model).get_extra_configs() + configs

    def train(self, tweets: List[Tweet]) -> None:
        def tweet_to_tuple(x):
            return (FreqDist(self.tokenizer(x.text)), x.emoji)

        # Generates tuples of all the tweets to form the corpus
        corpus = map(tweet_to_tuple, tweets)

        # Train this model!
        self.classif.train(corpus)

    def predict(self, text):
        return self.classif.classify(FreqDist(self.tokenizer(text)))
