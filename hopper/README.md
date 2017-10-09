# Hopper Models

This directory contains all the code used in our system (aka: **Hopper**).

## model.py

This file contains the classes and templates for the other models.

Classes:
- Tweet, this class contains the data for a given tweet. That is the text for the tweet and the id of the emoji linked to it.
- Model, this class is a parent class for every model and defines the interface which a model needs to implement.

## model_rand.py

This file contains the Random Model. This model returns a random emoji as its prediction.

## model_naive_bayes_baselines.py

There are two similar Naive Bayes models. Both use a bags of words filtering out infrequently used words.
One model uses a [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)
algorithm and the other uses a [Bernoulli Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes) algorithm.

For the sake of simplicity we will be using and explaining only the *Bernoulli Naive Bayes* Model. The reason for this decision is based on the fact that they very similar in design and much experimental research has found that the Bernoulli model gives better results for short text (in this case Tweets).

### Training
To train the Naive Bayes models we use in this project we follow a couple general steps. They are:
 * [Tokenization](#tokenization)
 * [*Bagination*](#bagination)
 * [Tf-idf transform](#tfidftransform)
 * [Select K best](#selectkbest)
 * finally the training the classifier, in our case the [Bernoulli Naive Bayes classifier](#train-bernoulli-naive-bayes-classifier).

#### Tokenization
The first step is to turn the

#### Bagination
This step is a step we created to transform the tokenized tweets in the form that the tf-idf transform needs.
A tokenized tweet is in the form of a list of strings which means the training corpus is a list of lists of strings. (Ex. ["I love u", "I love you but you hate me"] => [["i", "love", "u"], ["i", "love", "you", "but", "you", "hate", "me"]])


#### TfidfTransform

#### SelectKBest

#### Train Bernoulli Naive Bayes classifier

### Testing/Prediction
To test or make a prediction the model needs to prepare the tweet into a similar form which the training data was in.
We do the following steps:
 * [Tokenization](#tokenization)
 * [*Bagination*](#bagination)
 * the classify using the trained [Bernoulli Naive Bayes classifier](#use-bernoulli-naive-bayes-classifier)

#### Use Bernoulli Naive Bayes classifier
