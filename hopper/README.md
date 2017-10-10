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
Author: Jonathan Beaulieu  
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
The first step is to turn the plain tweet text given from the training data into a list of tokens which the model can be trained on. We decided to use the [**TweetTokenizer**](http://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.casual.TweetTokenizer) from the *nltk*.
We use the following settings:
- preserve_case=False
  - This means that all text is put into lowercase.
- reduce_len=True
  - This means that any string of repeating characters longer than 3 will be *cropped*. Ex. "!!!!" => ["!", "!", "!"], "yeeeeeeeessssss" => "yeeesss",
- strip_handles=True
  - This means that any tweeter handles will be removed completely. Ex. "I love @derpferd" => ["i", "love"]
  - Note: this means anything that matches the format of "@" followed by alphanumeric characters will be removed even if it is not a currently used handle.

The **TweetTokenizer** splits text into tokens based on whitespace and punctuation. Anything separated by whitespace is a token as well as any punctuation. One thing to note is that all punctuation characters are their own tokens except three dots (i.e. "..."). The tokenizer also provides the settings listed above.

Some examples:

| Input                      | Output                       |
|:---------------------------|:-----------------------------|
| `"I love u!"`              | `['i', 'love', 'u', '!']`    |
| `"I love @derpferd!"`      | `['i', 'love', '!']`         |
| `"Yeeeeeessssss!!!!!!!!!"` | `['yeeesss', '!', '!', '!']` |
| `"I love you but you hate me"`   | `['i', 'love', 'you', 'but', 'you', 'hate', 'me']`  |

#### Bagination
This step is a step we created to transform the tweets into the form that the tf-idf transform needs. We call this format a *bag*. A *bag* is a list of tuples where each tuple has a token as the first element and the count of occurrences as the second. Ex. `[("the", 3), ("dog", 2), ("cat", 1)]` would describe the string `"the dog the dog the cat"`. As you might have noticed bagination removes any positional data making a *bag* contains less information than a list of tokens.  
In the bagination step we convert the list of tweets into a list of tuples where the first element is a *bag* of the tweet text and the second element is the label the tweet contains. Ex. `Tweet<text: "the dog the dog the cat" emoji:1>` => `([("the", 3), ("dog", 2), ("cat", 1)], 1)`  
A list of these *bags* is what the next step takes.

#### TfidfTransform
Quote from Dennis's Documentation:
>TfidfTransformer transforms a count matrix to a normalized tf-idf representation. tf-idf means term-frequency times inverse document-frequency. The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

#### SelectKBest
The SelectKBest step, as it's name implies, selects the *k* best features. This step takes two arguments: a scoring function and *k*. The scoring function we use is [**cha2**](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2), really said chi-squared. Based on this SelectKBest selects the top *k* scoring tokens. We used a *k* value of 1000. We will need to do more research into if this value is optimal. The results from this selection feed into the next step, the classifier.

#### Train Bernoulli Naive Bayes classifier

Bernoulli Naive Bayes classifier is a type of Naive Bayes(NB) classifier. NB models all make the assumption that every feature is independent of every other feature. A main difference between Bernoulli NB model and other NB models is that Bernoulli works only with binary values for features (0 meaning the feature is not present and 1 meaning it is). We used the default settings which included Laplace smoothing and training class prior probabilities.

### Testing/Prediction
To test or make a prediction the model needs to prepare the tweet into a similar form which the training data was in.
We do the following steps:
 * [Tokenization](#tokenization)
 * [*Bagination*](#bagination)
   - Note: We use the same *Bagination* technique however we don't have the label.
 * the classify using the trained [Bernoulli Naive Bayes classifier](#applying-bernoulli-naive-bayes-classifier)

#### Applying Bernoulli Naive Bayes classifier
