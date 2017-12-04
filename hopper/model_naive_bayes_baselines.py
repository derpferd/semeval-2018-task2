""" Code Author: Jonathan Beaulieu
    Documentation Author: Dennis Asamoah Owusu

    Documentation References:
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
"""
from __future__ import division

from nltk.classify import SklearnClassifier
from nltk.probability import FreqDist
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from hopper.model_sklearn import SKLearnModel


class NaiveBayesModel(SKLearnModel):
    """This model classifies tweets into any one of twenty classes
    using NaiveBayes classification. The twenty classes each represents
    a different emoticon (e.g. love, crying). Putting a tweet in a class
    means that that tweet goes with the associated emoticon.
    """

    def __init__(self, classif_alg, k=1000):
        """Constructor

        Args:
            classif_alg: Which NavieBayes Classfier to use. E.g. Multinomial

        """

        """ Assigns a tokenizer function that converts a tweet into a list
        of tokens. reduce_len=True means that tokens that are repeated more
        than three times will be cut down to three. Note that ... is represented
        as one token having ... whereas !!! or *** will each be represented
        as three tokens having ! or * each.
        """
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                        reduce_len=True,
                                        strip_handles=True).tokenize

        """Creates a pipeline for use in classifying. Each argument will
        work on the data in sequence.

        The TfidTransformer normalizes the count matrix (in our case,
        the frequency of a word in each emoji class). Using the frequencies
        without normalization gives too much weight to words that occur
        very frequently in the corpus even though they are less informative features.
        The frequencies are normalized to a tf-idf represetnation. Tf-idf is
        term-frequency times inverse document-frequency.

        SelectKBest selects the best k features for use in classifying.
        """
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             # ('chi2', SelectKBest(chi2, k=k)),
                             ('nb', classif_alg())])
        self.classif = SklearnClassifier(pipeline)

    def train(self, tweets):
        """Trains the classifier.

        Args:
            tweets: a list of tweets to use in training the classifier.
        """

        def tweet_to_tuple(x):
            """Converts a tweet to a tuple.

            self.tokenizer is used to convert the tweet to a list of tokens.
            this list of tokens as well as the class of the tweet (contained in
            x.emoji) is passed to FreqDist which constructs the required tuple.

            Example: "I love you but you hate me" with x.emoji = 2

            tokenize: ["I", "love", "you", "but", "you", "hate", "me"]
            FreqDist: ([("I",1),("love",1),("you",2),("hate",1),("me",1)], 2)
            """
            return (FreqDist(self.tokenizer(x.text)), x.emoji)

        # Generates tuples of all the tweets to form the corpus
        corpus = map(tweet_to_tuple, tweets)

        self.classif.train(corpus)

    def predict(self, text):
        """Predicts what class a tweet should have

        Args:
            text: The tweet

        See self.train.tweet_to_tuple to see how FreqDist(self.tokenizer(text))
        works
        """
        return self.classif.classify(FreqDist(self.tokenizer(text)))


class MultinomialNaiveBayesModel(NaiveBayesModel):
    def __init__(self, k=1000):
        super().__init__(MultinomialNB, k)


class BernoulliNaiveBayesModel(NaiveBayesModel):
    def __init__(self, k=1000):
        super().__init__(BernoulliNB, k)
