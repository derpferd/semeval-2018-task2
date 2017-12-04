""" Code Author: Jonathan Beaulieu
"""
from __future__ import division

import json

import os
import shutil
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Conv1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import to_categorical

from hopper import Model


class CharNNModel(Model):
    """This model classifies tweets into any one of twenty classes
    using Neural Network classification.
    """
    def __init__(self, maxlen, **kargs):
        # Set the max number of characters to use from a tweet.
        self.maxlen = maxlen

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "maxlen", "default": 80},
                   {"name": "epochs", "default": 20},
                   {"name": "iteration_scoring", "default": True},
                   {"name": "checkpoint_saving", "default": True},
                   {"name": "max_non_improving_iterations", "default": 5}]
        return super(CharNNModel, CharNNModel).get_extra_configs() + configs

    def tokenize(self, text):
        return list(text)

    def create_model(self, maxlen, char_count, class_count):
        raise NotImplemented("Needs to be implemented.")

    def preprocess_data(self, tweets, test_dev_split):
        tokenized = []
        labels = []
        self.vocab = {"<unk>": 0}

        for tweet in tweets:
            part = []
            labels += [tweet.emoji]
            for c in tweet.text[:self.maxlen]:
                if c not in self.vocab:
                    self.vocab[c] = len(self.vocab)
                part += [self.vocab[c]]
            tokenized += [part]

        self.max_chars = len(self.vocab)
        self.class_count = len(set(labels))

        tokenized = sequence.pad_sequences(tokenized, maxlen=self.maxlen)

        labels = to_categorical(labels)

        split = int(test_dev_split * len(tweets))

        self.dev_set = tweets[split:]

        return (tokenized[:split], labels[:split]), (tokenized[split:], labels[split:])

    def process_test_data(self, texts):
        tokenized = []

        for text in texts:
            part = []
            for c in text:
                # if c is in the vocab get the value for it otherwise use the unknown char(0)
                part += [self.vocab.get(c, 0)]
            tokenized += [part]

        tokenized = sequence.pad_sequences(tokenized, maxlen=self.maxlen)

        return tokenized

    def train(self, tweets, continue_training=False, epochs=1):
        batch_size = 32
        test_dev_split = 0.9

        (x_train, y_train), (x_dev, y_dev) = self.preprocess_data(tweets, test_dev_split)

        if not continue_training:
            self.model = self.create_model(self.maxlen, self.max_chars, self.class_count)

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
        return [pred.argmax() for pred in preds]

    def save_model(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.h5"))
        obj = {"maxlen": self.maxlen,
               "vocab": self.vocab,
               "max_chars": self.max_chars,
               "class_count": self.class_count}
        json.dump(obj, open(os.path.join(path, "data.json"), "w"))

    def load_model(self, path):
        self.model = load_model(os.path.join(path, "model.h5"))
        obj = json.load(open(os.path.join(path, "data.json"), "r"))
        self.maxlen = obj["maxlen"]
        self.vocab = obj["vocab"]
        self.max_chars = obj["max_chars"]
        self.class_count = obj["class_count"]


class CharLSTMModel(CharNNModel):
    def __init__(self, maxlen, embedding_size=128, lstm_size=64, **kargs):
        super().__init__(maxlen)
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "embedding_size", "default": 128},
                   {"name": "lstm_size", "default": 64}]
        return super(CharBiLSTMModel, CharBiLSTMModel).get_extra_configs() + configs

    def create_model(self, maxlen, char_count, class_count):
        model = Sequential()
        model.add(Embedding(char_count, self.embedding_size, input_length=maxlen))
        model.add(LSTM(self.lstm_size))
        model.add(Dropout(0.5))
        model.add(Dense(class_count, activation="sigmoid"))

        model.compile('adam', 'categorical_crossentropy', metrics=["accuracy"])
        return model

    # def create_model(self, maxlen, char_count, class_count):
    #     model = Sequential()
    #     model.add(LSTM(128, input_shape=(maxlen, char_count)))
    #     model.add(Dense(char_count))
    #     model.add(Activation('softmax'))
    #
    #     optimizer = RMSprop(lr=0.01)
    #     model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    #
    #     return model


class CharBiLSTMModel(CharNNModel):
    def __init__(self, maxlen, embedding_size=128, lstm_size=64, **kargs):
        super().__init__(maxlen)
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size

    @staticmethod
    def get_extra_configs():
        configs = [{"name": "embedding_size", "default": 128},
                   {"name": "lstm_size", "default": 64}]
        return super(CharBiLSTMModel, CharBiLSTMModel).get_extra_configs() + configs

    def create_model(self, maxlen, char_count, class_count):
        model = Sequential()
        model.add(Embedding(char_count, self.embedding_size, input_length=maxlen))
        model.add(Bidirectional(LSTM(self.lstm_size)))
        model.add(Dropout(0.5))
        model.add(Dense(class_count, activation="sigmoid"))

        model.compile('adam', 'categorical_crossentropy', metrics=["accuracy"])
        return model


class CharCNNModel(CharNNModel):
    def create_model(self, maxlen, char_count, class_count):
        pass


class CharLSTMCNNModel(CharNNModel):
    def __init__(self, maxlen, embedding_size=128, lstm_size=64, kernel_size=5, filters=64, pool_size=4, **kargs):
        super().__init__(maxlen)
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size

    @staticmethod
    def get_extra_configs():
        # TODO: add batch_size
        configs = [{"name": "embedding_size", "default": 128},
                   {"name": "lstm_size", "default": 64},
                   {"name": "kernel_size", "default": 5},
                   {"name": "filters", "default": 64},
                   {"name": "pool_size", "default": 4}]
        return super(CharBiLSTMModel, CharBiLSTMModel).get_extra_configs() + configs

    def create_model(self, maxlen, char_count, class_count):
        model = Sequential()
        model.add(Embedding(char_count, self.embedding_size, input_length=maxlen))
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


class CharBiLSTMCNNModel(CharNNModel):
    def __init__(self, maxlen, embedding_size=128, lstm_size=64, kernel_size=5, filters=64, pool_size=4, **kargs):
        super().__init__(maxlen)
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size

    @staticmethod
    def get_extra_configs():
        # TODO: add batch_size
        configs = [{"name": "embedding_size", "default": 128},
                   {"name": "lstm_size", "default": 64},
                   {"name": "kernel_size", "default": 5},
                   {"name": "filters", "default": 64},
                   {"name": "pool_size", "default": 4}]
        return super(CharBiLSTMModel, CharBiLSTMModel).get_extra_configs() + configs

    def create_model(self, maxlen, char_count, class_count):
        model = Sequential()
        model.add(Embedding(char_count, self.embedding_size, input_length=maxlen))
        model.add(Dropout(0.25))
        model.add(Conv1D(self.filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(Bidirectional(LSTM(self.lstm_size)))
        model.add(Dense(class_count))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
