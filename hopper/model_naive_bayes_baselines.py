from __future__ import division
from hopper import Model
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


class NaiveBayesModel(Model):
    """This model classifies tweets into any one of twenty classes
    using NaiveBayes classification. The twenty classes each represents
    a different emoticon (e.g. love, crying). Putting a tweet in a class
    means that that tweet goes with the associated emoticon.
    """
    def __init__(self, classif_alg):
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
        self.labels = list(range(20))

        """Creates a pipeline for use in classifying. Each argument will
        work on the data in sequence.

        TfidfTransformer transforms a count matrix
        to a normalized tf-idf representation. tf-idf means term-frequency
        times inverse document-frequency. The goal of using tf-idf instead of
        the raw frequencies of occurrence of a token in a given document is
        to scale down the impact of tokens that occur very frequently in a
        given corpus and that are hence empirically less informative than
        features that occur in a small fraction of the training corpus.

        SelectKBest selects the best k features for use in classifying.
        """
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('chi2', SelectKBest(chi2, k=1000)),
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
    def __init__(self):
        super().__init__(MultinomialNB)


class BernoulliNaiveBayesModel(NaiveBayesModel):
    def __init__(self):
        super().__init__(BernoulliNB)
