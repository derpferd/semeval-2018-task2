# Hopper Models

This directory contains all the code used in our system (aka: **Hopper**).

## model.py

This file contains the classes and templates for the other models.

Classes:
- Tweet, this class contains the data for a given tweet. That is the text for the tweet and the id of the emoji linked to it.
- Model, this class is a parent class for every model and defines the interface which a model needs to implement.

## model_rand.py

This file contains the Random Model. This model returns a random emoji as its prediction.

## model_most_frequent_class.py
Author: Dennis Asamoah Owusu

This class contains the MostFrequentClassModel which assigns each tweet to the MostFrequentClass.
It simply extracts the emoji associated with each tweet in the training data, determines which one appears the most and uses that as the emoji for every tweet it predicts.

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

Bernoulli Naive Bayes classifier is a type of Naive Bayes(NB) classifier. NB models all make the assumption that every feature is independent of every other feature. A main difference between Bernoulli NB model and other NB models is that Bernoulli works only with binary values for features (0 meaning the feature is not present and 1 meaning it is) and not only looks at the tokens in a given *document*(See [Definitions](#definitions) for details) but also the token that are not in the *document*. We used the default settings which included Laplace smoothing and training class prior probabilities.

Naive Bayes needs to train two parts: the priors, the probability of a class given a *document*, and the probabilities of each token given each class.
The Bernoulli Naive Bayes Model estimates the probability of a token given a class as the percentage of *documents* in the class which contain the token. As you can see the frequency of a token in a *document* is not considered. Also note that during the training phase a *vocab* is created. To read about how a class prediction is made see the [Applying Bernoulli Naive Bayes classifier](#applying-bernoulli-naive-bayes-classifier) section below.

### Testing/Prediction
To test or make a prediction the model needs to prepare the tweet into a similar form which the training data was in.
We do the following steps:
 * [Tokenization](#tokenization)
 * [*Bagination*](#bagination)
   - Note: We use the same *Bagination* technique however we don't have the label.
 * the classify using the trained [Bernoulli Naive Bayes classifier](#applying-bernoulli-naive-bayes-classifier)


#### Applying Bernoulli Naive Bayes classifier

With the probabilities for each class, priors, and the probabilities of a token given a class (Note: for unknown tokens Laplace smoothing is applied) the classifier can now calculate a probability for each class given a *document*.
For each class the classifier follows the following algorithm to calculate the probability of the class given the *document*, then picks the class with the greatest probability.

##### Class Probability Algorithm
 - Set the score to the log of the prior of the class being evaluated
 - For each token in the *vocab*
   - if the token is in the *document* add the log of the probability of a token given a class to the score
   - else add the log of one minus the probability of a token given a class to the score

After this the scores for each class can be compared to each other. The largest is returned as the predicted class.

### Definitions

**Document:** A single unit of text which represents a single class. Note a class have have many documents. In our case each tweet is a document.

**Bag:** A set of words and their frequencies. Note there is no positional data.

**Vocab:** A set of every token that is seen during the training phase. Note we say that the length of the *vocab* is one longer than the number of tokens it contains since we are using Laplace(add-1) smoothing.

### References

1. C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
Information Retrieval. Cambridge University Press, pp. 234-265.
http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html or https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf

2. SciKit Learn - Documentation http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes
