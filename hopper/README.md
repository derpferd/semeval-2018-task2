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
