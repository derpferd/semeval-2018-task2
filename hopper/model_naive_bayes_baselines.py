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
    def __init__(self, classif_alg):
        self.tokenizer = TweetTokenizer(preserve_case=False,
                                        reduce_len=True,
                                        strip_handles=True).tokenize
        self.labels = list(range(20))
        pipeline = Pipeline([('tfidf', TfidfTransformer()),
                             ('chi2', SelectKBest(chi2, k=1000)),
                             ('nb', classif_alg())])
        self.classif = SklearnClassifier(pipeline)

    def train(self, tweets):
        def tweet_to_tuple(x):
            return (FreqDist(self.tokenizer(x.text)), x.emoji)

        corpus = map(tweet_to_tuple, tweets)
        self.classif.train(corpus)

    def predict(self, text):
        return self.classif.classify(FreqDist(self.tokenizer(text)))


class MultinomialNaiveBayesModel(NaiveBayesModel):
    def __init__(self):
        super().__init__(MultinomialNB)


class BernoulliNaiveBayesModel(NaiveBayesModel):
    def __init__(self):
        super().__init__(BernoulliNB)
