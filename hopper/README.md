# Hopper Models

This directory contains all the code used in our system (aka: **Hopper**).

## model.py

This file contains the classes and templates for the other models.

Classes:
- Tweet, this class contains the data for a given tweet. That is the text for the tweet and the id of the emoji linked to it.
- Model, this class is a parent class for every model and defines the interface which a model needs to implement.

## model_rand.py

This file contains the Random Model. This model returns a random emoji as its prediction.

## model_unigram_baseline.py

This model will use a language model per emoji and train each on the tweets given for that emoji. To predict the model will choose the emoji who's language model returns the highest probability.
