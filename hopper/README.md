# Hopper Models
Multiple Authors. The model creator is the author of the section about the model. They are list under each section.

This directory contains all the code used in our system (aka: **Hopper**).

## model.py
Author: Jonathan Beaulieu

This file contains the classes and templates for the other models.

Classes:
- Tweet, this class contains the data for a given tweet. That is the text for the tweet and the id of the emoji linked to it.
- Model, this class is a parent class for every model and defines the interface which a model needs to implement.

## model_sklearn.py
Author: Jonathan Beaulieu

This file contains an abstract class which implements the `save_model` and `load_model` methods for any sklearn model.

## model_rand.py
Author: Jonathan Beaulieu

This file contains the Random Model. This model returns a random emoji as its prediction.

## model_most_frequent_class.py
Author: Dennis Asamoah Owusu

This class contains the MostFrequentClassModel which assigns each tweet to the MostFrequentClass.
It simply extracts the emoji associated with each tweet in the training data, determines which one appears the most and uses that as the emoji for every tweet it predicts.

## model_naive_bayes_baselines.py
Author: Jonathan Beaulieu

Note before you read: Words in *italics* mean that the word is defined in the [Definitions](#definitions) section.

There are two similar Naive Bayes models. Both use a bags of words filtering out infrequently used words.
One model uses a [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)
algorithm and the other uses a [Bernoulli Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes) algorithm.

For the sake of simplicity we will be using and explaining only the **Bernoulli Naive Bayes** Model. The reason for this decision is based on the fact that they very similar in design and much experimental research has found that the Bernoulli model gives better results for short text (in this case Tweets) ([2](#references)).

### Training
To train the Naive Bayes models we use in this project we follow a couple general steps. They are:
 * [Tokenization](#tokenization)
 * [**Bagination**](#bagination)
 * [Tf-idf transform](#tfidftransform)
 * [Select K best](#selectkbest)
 * finally the training the classifier, in our case the [Bernoulli Naive Bayes classifier](#train-bernoulli-naive-bayes-classifier).

#### Tokenization
The first step is to turn the plain tweet text given from the training data into a list of tokens which the model can be trained on. We decided to use the [**TweetTokenizer**](http://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.casual.TweetTokenizer) from the **nltk**.
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
> The TfidTransformer normalizes the count matrix (in our case, the frequency of a word in each emoji class). Using the frequencies without normalization gives too much weight to words that occur very frequently in the corpus even though they are less informative features. The frequencies are normalized to a tf-idf representation. Tf-idf is term-frequency times inverse document-frequency.

#### SelectKBest
The SelectKBest step, as it's name implies, selects the *k* best features (features is another word for token). This step takes two arguments: a scoring function and *k*. The scoring function is a function which assigns a score to each feature based on some of its properties. We use is [**chi2**](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2) (said chi-squared). This function assigns scores using the chi-squared test between feature frequencies and the classes. The chi-squared test measures the dependence between random variables, resulting in features that are independent of class receiving a lower score. An example of this is the frequency of the token `the` which probability exhibits a random distribution between classes, giving it a very low chi2 score.
Based on the scores, SelectKBest selects the top *k* scoring tokens. We used a *k* value of 1000 since that is the default value used in SKLearn's example ([2](#references)). In the future the *k* value will be set based on a development set. The results from this selection feed into the next step, the classifier.

#### Train Bernoulli Naive Bayes classifier

Bernoulli Naive Bayes classifier is a type of Naive Bayes(NB) classifier. NB models all make the assumption that every feature is independent of every other feature. A main difference between Bernoulli NB model and other NB models is that Bernoulli works only with binary values for features (0 meaning the feature is not present and 1 meaning it is) and not only looks at the tokens in a given *document* but also the token that are not in the *document*. We used the default settings which included Laplace smoothing and training class *priors*.

Naive Bayes needs to train two parts: the *priors* and *conditional probability*(condprob).
The Bernoulli Naive Bayes Model estimates the *condprob* as the percentage of *documents* in the class which contain the token. As you can see the frequency of a token in a *document* is not considered. Also note that during the training phase a *vocab* is created. To read about how a class prediction is made see the [Applying Bernoulli Naive Bayes classifier](#applying-bernoulli-naive-bayes-classifier) section below.

**Equations**  
The following equations describe the above description of how to train a Bernoulli Naive Bayes classifier.  
let `V` be the *vocab*.  
let `N` be the total number of *documents*.  
let `N_c` be the number of *documents* in class `c`.  
let `N_tc` be the number of *documents* in class `c` containing the token `t`.  
`prior[c] = N_c/N`  
`condprob[t][c] = (N_tc + 1)/(N_c + 2)`

### Testing/Prediction
To test or make a prediction the model needs to prepare the tweet into a similar form which the training data was in.
We do the following steps:
 * [Tokenization](#tokenization)
 * [**Bagination**](#bagination)
   - Note: We use the same **Bagination** technique however we don't have the label.
 * the classify using the trained [Bernoulli Naive Bayes classifier](#applying-bernoulli-naive-bayes-classifier)


#### Applying Bernoulli Naive Bayes classifier

With the *priors* and the probabilities of a token given a class (Note: for unknown tokens Laplace smoothing is applied) the classifier can now calculate a probability for each class given a *document*.
For each class the classifier follows the following algorithm to calculate the probability of the class given the *document*, then picks the class with the greatest probability.

**Class Probability Algorithm**
 - Set the score to the log of the *prior* of the class being evaluated
 - For each token in the *vocab*
   - if the token is in the *document* add the log of the probability of a token given a class to the score
   - else add the log of one minus the probability of a token given a class to the score

After this the scores for each class can be compared to each other. The largest is returned as the predicted class.

**Equations**  
The following equations describe the above description of how to apply a Bernoulli Naive Bayes classifier.  
given `d` a *document*  
let `prior[c]` be the function as describe in the training section  
let `condprob[t][c]` be the function as describe in the training section  
start with `score[c] = log(prior[c])`  
for each `t` in `V` {  
if `t` is in `d`: `score[c] += log(condprob[t][c])`  
else: `score[c] += log(1 - condprob[t][c])`  
}  
return `argmax score[c]`


### Definitions

**Document:** A single unit of text which represents a single class. Note a class have have many documents. In our case each tweet is a document.

**Bag:** A set of words and their frequencies. Note there is no positional data.

**Vocab:** A set of every token that is seen during the training phase. Note we say that the length of the *vocab* is one longer than the number of tokens it contains since we are using Laplace(add-1) smoothing.

**Prior:** The probability of a class given a *document*.

**Conditional probability (CondProb):** the probabilities of a token given a class.

**Crop:** means to remove.

### References

1. C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
Information Retrieval. Cambridge University Press, pp. 234-265.
http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html or https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf

2. SciKit Learn - Documentation http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes

## model_char_lstm.py
Author: Jonathan Beaulieu

This model file contains many different character based neural network classifier models. All of the models use Keras as a backend.
It contains the following classes (which contain a different model except the first one.):  
 - CharNNModel
   - This is the base class for all of the other models in this file.
   - It contains the following notable methods:
     - `preprocess_data`
       - This function tokenizes the tweets into an array of numbers using a mapping stored in a vocab. It also turns the labels into one hot encoded vectors. Lastly, it splits the data into a train set and a dev set. Basically this method makes sure that the data is already to be used in the training step.
     - `train`
       - This function handles the training so all the other models can use this code instead of rewriting it.
     - `predict`
       - This function uses the model created by `train` to predict the class based on the tweet text.
 - CharLSTMModel
 - CharBiLSTMModel
 - CharBiLSTMCNNModel

All of our Neural Network Models have a of things in common. I will discuss these things here.

#### Early Stopping
When it comes to NN Models over-fitting is a very real issue. This is mostly due to the fact most if not all NN Models are trained using many iterations(aka. epochs). As the model gets trained more and more on the train data the score keeps increasing however at some point the model will be fit so closely to the training data it will not work well with other data of the same type. To prevent this we use a **validation set**. This data set is removed from the training dataset and used to score the model after each training epoch. After each epoch if the model stops improving the score for the validation data we stop the training. Note: The size of the validation set is hard coded to 10% of the training set (should make this configurable in the future however for now a 90/10 split is pretty standard so no change needed).

In our case we have a configuration value, `max_non_improving_iterations`, which controls how many iterations to train without them improving the score against the validation set. The default value is 5. This allows the model's score to go down some for an iteration when it might go much higher in the next.

#### Layers
All our NN Models have a common structure. Each one is a sequential model made up of **layers**. Each layer's output is the next ones input. The first layer takes two values. First is an array of vectors, where each vector is a list of integers which represent the tokens in the tweet text (ex. char-based "speech" -> [1,2,3,3,4,5]). Note this vector is padded such that all the vectors have the same length. This length is configured in the `maxlen` variable. The other parameter is the "answers" in our case this is an array of one-hot encoded vectors, where each vector represents the class/label (e.g. emoji) which belongs to the matching text.

#### Dropout Layer
Another technique we use to prevent over-fitting our models is a Dropout layer. In this layer some of the learned data is "thrown away". This is done by randomly removing the values(aka. weights) form nodes in the model which also removes the connections to that node at the same time. This is the most popular form of regularization (preventing over-fitting and making the model more general) in NN Models.

#### Dense Layer
This layer is a densely-connected NN. We use this layer to "transform" the high dimensional vectors coming out of the LSTM to the dimensions that match our output (the one-hot encoded labels).

#### Embedding Layer
For all of out char-based NN models the first layer is a trainable character embedding layer. This layer creates a unique vector to represent each token (note this adds a dimension, so now the data is an array of vectors where the vectors are embeddings, which are vectors themselves). Each vector is randomly assigned at first however during the training phase it is modified based the output of the system compared to the correct values. The size of these embeddings are set based on the `embedding_size` configuration, which defaults to 128 (experimentally found to be the based for our use case).

Note: Below we will talk about the layers in the models.

### CharLSTMModel
This model is a "plain" lstm model.
This model has four layers:
 - Embedding (This is discussed in the section above.)
 - LSTM
   - This is a type of RNN, which means it's input and output layer are connected by multiple hidden layers.
   - The hidden layer values are updated after each based on the difference between the models output and the expected output.
 - Dropout (This is discussed in the section above.)
 - Dense (This is discussed in the section above.)


### CharBiLSTMModel
This model was inspired by the work by in *Are emojis predictable?* [1]. Their best model was a Character-based Bidirectional (Long Short Term Memory) LSTM Model.

 - Embedding (This is discussed in the section above.)
 - Bidirectional LSTM
   - This just like the LSTM except it trains itself on each input twice once normally and once with the input reversed.
 - Dropout (This is discussed in the section above.)
 - Dense (This is discussed in the section above.)

Note: This system shows slight improvement over our LSTM system.

### CharBiLSTMCNNModel
We created this model based on the above CharBiLSTMModel and the concepts we learned from Cicero dos Santos and Maira Gatti's paper [2].

 - Embedding (This is discussed in the section above.)
 - Convolutional Layer (CNN)
   - This layer uses a Rectifier activation function to "merge" a window of inputs into a single input. The window size is configurable through the `kernel_size` variable. This is helps the model since each single input(a char) also includes some contextual information through this process.
 - Bidirectional LSTM (This is discussed in the section above.)
 - Dropout (This is discussed in the section above.)
 - Dense (This is discussed in the section above.)

### References

1. Francesco Barbieri, Miguel Ballesteros, and Horacio Saggion. 2017. Are emojis predictable? In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. Association for Computational Linguistics, pages 105–111. http://www.aclweb.org/anthology/E17-2017.
2. Cicero dos Santos and Maira Gatti. 2014. Deep convolutional neural networks for sentiment analysis of short texts. In Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers. Dublin City University and Association for Computational Linguistics, Dublin, Ireland, pages 69–78. http://www.aclweb.org/anthology/C14-1008.

## model_word_nn.py
Author: Jonathan Beaulieu

### WordEmbeddingCNNModel
This model is the same as CharBiLSTMCNNModel exact instead of char-based embedding we use pretrained word based embeddings. We got these from the [Glove](https://nlp.stanford.edu/projects/glove/). We use the 100d twitter vectors. The results were very poor from this method so we decided to focus on other methods instead.

## model_svm.py
Author: Jonathan Beaulieu

Quoted from Dennis:
> [Linear Support Vector Machine] LSVMs are by default binary classifiers. Each document to be classified is represented as a vector based on the features of the document. In our case, the features was the words making up the document (Bag of words). The document turned feature vector is then represented in some n-dimensional vector space where n is the dimensions of the feature vectors.

> Having represented these vectors, the Support Vector machine learns a hyperplane that separates the vectors based on the class. Ideally, you want all the vectors belonging to the first class to be, say, above the hyperplane and all the vectors belonging to the second class to be below the hyperplane. Each new document to be classified is represented in the vector space and the class it belongs to depends on whether it is below or above the plane. The way the separating hyperplane is drawn (e.g. slope) is determined by vectors close to it known as support vectors. Specifically, the hyperplane is drawn so as to maximize the distance between it and these support vectors. This is a simplification of how LSVMs work but sufficient to show that it can be used with a Bag of Words feature to classify documents into one of two classes. The LSVM binary classifier can be employed in different ways to perform a multi-classification and, in our case, we used a one-vs-rest strategy for doing multi-classification. This means the multi-classification is broken up into many binary-classifications, one for each class. Each binary-classification decides whether the input belongs more in the class or one of the rest of the classes. We compare all the results and pick the class with the highest probability

To create a Support Vector Machine Model we used the SKLearn Library.
We used a pipeline similar to the one we used for our Baseline Model.
Our pipeline looks like:
 - Tokenization
 - TfidfTransform
 - train SVM classifier

Tokenization is done as described in the [Tokenization](#tokenization) section. Likewise with the [TfidfTransform](#tfidftransform).

In the training phase of Pipeline we use SKLearn Library to train the classifier. We use a one-vs-rest strategy for doing multi-classification as mentioned by Dennis. To understand how this step works please read Dennis' description above.

### References
https://machinelearningmastery.com/support-vector-machines-for-machine-learning/  
https://nlp.stanford.edu/IR-book/html/htmledition/multiclass-svms-1.html  
https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-the-linearly-separable-case-1.html  
