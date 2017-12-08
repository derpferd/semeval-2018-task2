""" Code Author: Jonathan Beaulieu
"""
from __future__ import division

import json

import os
import shutil
from keras import Input
from keras.engine import training
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D
from keras.models import load_model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np

from hopper import Model


GLOVE_DIR = os.path.join('data', 'glove.twitter')


class WordNNModel(Model):
    """This model classifies tweets into any one of twenty classes
    using Neural Network classification on word units.
    """
    def __init__(self, maxlen, maxwords, embedding_size, **kargs):
        self.maxlen = maxlen
        self.maxwords = maxwords
        self.embedding_size = embedding_size
        self.tokenizer = Tokenizer(num_words=self.maxwords)
        self.load_embeddings()

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "maxlen", "default": 20},
                   {"name": "maxwords", "default": 20000},
                   {"name": "epochs", "default": 20},
                   {"name": "iteration_scoring", "default": True},
                   {"name": "checkpoint_saving", "default": True},
                   {"name": "max_non_improving_iterations", "default": 5},
                   {"name": "embedding_size", "default": 100}]
        return super(WordNNModel, WordNNModel).get_extra_configs() + configs

    def create_model(self, maxlen, num_words, class_count):
        raise NotImplemented("Needs to be implemented.")

    def load_embeddings(self):
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

    def preprocess_data(self, tweets, test_dev_split):
        tokenized = []
        labels = []

        for tweet in tweets:
            labels += [tweet.emoji]
            tokenized += [tweet.text]

        self.tokenizer.fit_on_texts(tokenized)
        word_index = self.tokenizer.word_index
        tokenized = self.tokenizer.texts_to_sequences(tokenized)
        tokenized = pad_sequences(tokenized, maxlen=self.maxlen)

        self.class_count = len(set(labels))

        labels = to_categorical(labels)

        split = int(test_dev_split * len(tweets))

        self.dev_set = tweets[split:]

        # prepare embedding matrix
        self.num_words = min(self.maxwords, len(word_index))
        self.embedding_matrix = np.zeros((self.num_words, self.embedding_size))
        for word, i in word_index.items():
            if i >= self.maxwords:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        return (tokenized[:split], labels[:split]), (tokenized[split:], labels[split:])

    def process_test_data(self, texts):
        tokenized = self.tokenizer.texts_to_sequences(texts)
        tokenized = pad_sequences(tokenized, maxlen=self.maxlen)

        return tokenized

    def train(self, tweets, continue_training=False, epochs=1):
        batch_size = 32
        test_dev_split = 0.9

        (x_train, y_train), (x_dev, y_dev) = self.preprocess_data(tweets, test_dev_split)

        if not continue_training:
            self.model = self.create_model(self.maxlen, self.num_words, self.class_count)

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_dev, y_dev))

    def predict(self, text):
        tests = self.process_test_data([text])
        preds = self.model.predict(tests, verbose=0)[0]
        return preds.argmax()

    def batch_predict(self, texts):
        tests = self.process_test_data(texts)
        preds = self.model.predict(tests, verbose=0)

    def save_model(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.h5"))

    def load_model(self, path):
        self.model = load_model(os.path.join(path, "model.h5"))

    def tokenize(self, text):
        return self.tokenizer.texts_to_sequences(texts)


class WordEmbeddingCNNModel(WordNNModel):
    def __init__(self, maxlen, maxwords, embedding_size, lstm_size, kernel_size, filters, pool_size, activation, optimizer, **kargs):
        super().__init__(maxlen, maxwords, embedding_size)
        assert embedding_size in [25, 50, 100, 200]
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.activation = activation
        self.optimizer = optimizer

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "lstm_size", "default": 64},
                   {"name": "kernel_size", "default": 5},
                   {"name": "filters", "default": 64},
                   {"name": "pool_size", "default": 4},
                   {"name": "activation", "default": "sigmoid"},
                   {"name": "optimizer", "default": "adam"}]
        return super(WordEmbeddingCNNModel, WordEmbeddingCNNModel).get_extra_configs() + configs

    def create_model(self, maxlen, num_words, class_count):
        model = Sequential()
        model.add(Embedding(num_words,
                            self.embedding_size,
                            weights=[self.embedding_matrix],
                            input_length=self.maxlen,
                            trainable=False))
        model.add(Dropout(0.25))
        model.add(Conv1D(self.filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_size))
        model.add(Dense(class_count))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
