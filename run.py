""" Code Author: Jonathan Beaulieu"""

# Purpose: Trains and Tests the models developed and prints out the result file for the respective input based on a
# config file passed in as a cli argument.
from __future__ import print_function

import json
from socket import gethostname

import os
import time
import argparse
from hopper import Tweet
from hopper.confusion_matrix import ConfusionMatrix
from hopper.scorer import Scorer
from hopper.model_rand import RandModel
from hopper.model_naive_bayes_baselines import BernoulliNaiveBayesModel
from hopper.model_most_frequent_class import MostFrequentClassModel
from hopper.model_svm import LinearSVCModel

# DEBUG_NN_MODELS = True
MACHINE_NAME = gethostname()
VERBOSE = True

NON_NN_MODEL_CLS = [RandModel(),
                    MostFrequentClassModel(),
                    BernoulliNaiveBayesModel(),
                    LinearSVCModel(),
                    ]
NON_NN_MODELS = dict(zip(map(lambda x: x.__class__.__name__, NON_NN_MODEL_CLS), NON_NN_MODEL_CLS))

# if not os.path.exists("models"):
#     os.mkdir("models")
# if not os.path.exists(os.path.join("models", MACHINE_NAME)):
#     os.mkdir(os.path.join("models", MACHINE_NAME))


def count(data, label):  # Function to count the data available for the respective test/train class. Gives out the number of tweets available for each gold emoji in testing and training data
    return sum([1 for t in data if t.emoji == label])


# There are 2 parts of data that are being handled, .text file with the tweets listed out and .labels file which has a corresponding label(emoji) assigned to each and every tweet in the text file.
# The function returns the tweets as the output which a list of text and their corresponding emoji label.
def load_tweets(basepath):  # Loads the twitter data file and respective labels file .
    text_path = basepath + ".text"
    labels_path = basepath + ".labels"

    # Read in input
    try:
        text_fp = open(text_path, 'r')
        text = text_fp.readlines()
        text_fp.close()
    except IOError:
        print("Had error reading from: {}".format(text_path))
    try:
        labels = open(labels_path, 'r').readlines()  # noqa
    except IOError:
        print("Had error reading from: {}".format(text_path))

    tweets = []  # type: List[Tweet]
    for text, label in zip(text, labels):
        tweets += [Tweet(text, int(label))]

    return tweets


class Config(object):
    def __init__(self, id, model, json_obj=None):
        self.id = id.replace(" ", "_")
        self.model = model
        self.json_obj = json_obj
        self.extras = {}

    def __str__(self):
        return "<Config id:{}>".format(self.id)

    def parse_extra_config(self, extra_config):
        for cfg in extra_config:
            name = cfg["name"]
            default = cfg.get("default", None)
            val = self.json_obj.get(name, default)
            if val is not None:
                setattr(self, name, val)
                self.extras[name] = val
            else:
                raise ValueError("Could not find '{}' in config".format(name))

    @classmethod
    def from_json_obj(cls, obj):
        config_id = obj["id"]
        model = obj["model"]
        config = cls(config_id, model, obj)
        config.language = obj.get("language", "us")
        config.data_type = obj.get("data_type", "trial")
        config.folds = obj.get("folds", 1)
        config.confusion_matrix = obj.get("confusion_matrix", True)
        return config


def get_log_file(config_id, recover=False):
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists(os.path.join("output", config_id)):
        os.mkdir(os.path.join("output", config_id))

    if recover:
        return open(os.path.join("output", config_id, "{}.log".format(MACHINE_NAME)), "a")
    else:
        return open(os.path.join("output", config_id, "{}.log".format(MACHINE_NAME)), "w")


def log_checkpoint(config_id, fold, iteration, model_path):
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists(os.path.join("output", config_id)):
        os.mkdir(os.path.join("output", config_id))

    obj = {"fold": fold,
           "iteration": iteration,
           "model_path": model_path}

    json.dump(obj, open(os.path.join("output", config_id, "{}_checkpoint.json".format(MACHINE_NAME)), "w"))


def get_checkpoint(config_id):
    checkpoint_path = os.path.join("output", config_id, "{}_checkpoint.json".format(MACHINE_NAME))
    if os.path.exists(checkpoint_path):
        return json.load(open(checkpoint_path, "r"))
    else:
        return None


def get_test_train_sets(tweets, cur_fold, folds):
    # Calculate the number of tweets that will be in the test set
    test_size = int(len(tweets) / folds)
    # Get the section of tweets into the test set
    test_data = tweets[cur_fold * test_size: (cur_fold + 1) * test_size]

    # create the training set from all but the tweets that are in the test set.
    train_data = []
    if cur_fold > 0:
        train_data += tweets[0: cur_fold * test_size]
    if cur_fold + 1 < folds:
        train_data += tweets[(cur_fold + 1) * test_size:]

    return train_data, test_data


def get_class_count(tweets):
    return len(set([t.emoji for t in tweets]))


def run_non_nn_model(config):
    # Get the model object from the config string.
    model = NON_NN_MODELS[config.model]
    # Get the log file so we can use it.
    log = get_log_file(config.id)

    # Load tweets
    if VERBOSE:
        print("Loading tweets...", file=log, flush=True)
    tweets = load_tweets(os.path.join("data", config.data_type, config.language + "_" + config.data_type))

    # Get the number of classes
    class_count = get_class_count(tweets)

    # Set up a scorer/confusion matrix.
    if config.confusion_matrix:
        total_scorer = ConfusionMatrix(class_count)
    else:
        total_scorer = Scorer()

    # Do fold number of cross folds
    if VERBOSE:
        print("Doing {} cross folds...".format(config.folds), file=log, flush=True)
    for i in range(config.folds):
        # Get the data sets
        if VERBOSE:
            print("Loading Data...", file=log, flush=True)
        train_data, test_data = get_test_train_sets(tweets, i, config.folds)

        # Train the model
        if VERBOSE:
            print("Training Model...", file=log, flush=True)
        model.train(train_data)

        # Score this fold.
        # Get a fold scorer/confusion matrix.
        if config.confusion_matrix:
            fold_scorer = ConfusionMatrix(class_count)
        else:
            fold_scorer = Scorer()

        predictions = model.batch_predict([tweet.text for tweet in test_data])
        for prediction, gold in zip(predictions, [tweet.emoji for tweet in test_data]):
            fold_scorer.add(gold, prediction)
            total_scorer.add(gold, prediction)

        # Print out the results
        print("\n----- Results -----\n{}\n".format(fold_scorer.get_score()), file=log, flush=True)

        if VERBOSE:
            # Print out the result details
            print("--- Details ---", file=log, flush=True)
            print("Training data len: {} Testing data len: {}".format(len(train_data), len(test_data)), file=log, flush=True)
            print("Training data class counts: " + ", ".join([str(i) + ": " + str(count(train_data, i)) for i in range(class_count)]), file=log, flush=True)
            print("Testing  data class counts: " + ", ".join([str(i) + ": " + str(count(test_data, i)) for i in range(class_count)]), file=log, flush=True)

        # Print out the confusion matrix if enabled
        if config.confusion_matrix:
            print("\n--- Matrix ---\n{}\n".format(fold_scorer), file=log, flush=True)

    print("\n----- Results for all folds -----\n{}\n".format(total_scorer.get_score()), file=log, flush=True)
    if config.confusion_matrix:
        print("--- Matrix ---\n{}\n".format(total_scorer), file=log, flush=True)

    log.close()


def run_nn_model(config, recover=False):
    # Get the log file so we can use it.
    log = get_log_file(config.id, recover)
    start_fold = 0
    start_iteration = 0
    checkpoint = None
    if recover:
        checkpoint = get_checkpoint(config.id)
        if checkpoint:
            start_fold = checkpoint["fold"]
            start_iteration = checkpoint["iteration"] + 1
            print("Recovering from fold {}, iteration {}".format(start_fold, start_iteration), file=log, flush=True)

    from hopper.model_char_lstm import CharLSTMModel, CharBiLSTMModel, CharLSTMCNNModel
    model_clss = [CharLSTMModel, CharBiLSTMModel, CharLSTMCNNModel]
    name_to_model_cls = dict(zip(map(lambda x: x.__name__, model_clss), model_clss))

    # Get model
    model_cls = name_to_model_cls[config.model]
    extras = model_cls.get_extra_configs()
    config.parse_extra_config(extras)
    print("Extra Configs: {}".format(config.extras), file=log, flush=True)
    model = model_cls(**config.extras)
    if recover and checkpoint:
        model.load_model(checkpoint["model_path"])

    # Load tweets
    if VERBOSE:
        print("Loading tweets...", file=log, flush=True)
    tweets = load_tweets(os.path.join("data", config.data_type, config.language + "_" + config.data_type))

    # Get the number of classes
    class_count = get_class_count(tweets)

    # Set up a scorer/confusion matrix.
    if config.confusion_matrix:
        total_scorer = ConfusionMatrix(class_count)
    else:
        total_scorer = Scorer()

    # If the epochs is -1 then that means we need to run till the model stops improving.
    if config.epochs == -1:
        # TODO: add support for saving and loading
        config.iteration_scoring = True
        config.epochs = 99999

    # Do fold number of cross folds
    if VERBOSE:
        print("Doing {} cross folds...".format(config.folds), file=log, flush=True)
    for fold in range(start_fold, config.folds):
        # Get the data sets
        if VERBOSE:
            print("Loading Data...", file=log, flush=True)
        train_data, test_data = get_test_train_sets(tweets, fold, config.folds)

        # Train the model
        if VERBOSE:
            print("Training Model...", file=log, flush=True)

        if config.iteration_scoring or config.checkpoint_saving:
            stop = False
            max_iteration_score = 0
            iterations_since_max = 0
            for iteration in range(start_iteration, config.epochs):
                model.train(train_data, continue_training=iteration!=0, epochs=1)
                if config.iteration_scoring:
                    if config.confusion_matrix:
                        iteration_scorer = ConfusionMatrix(class_count)
                    else:
                        iteration_scorer = Scorer()
                    predictions = model.batch_predict([tweet.text for tweet in test_data])
                    for prediction, gold in zip(predictions, [tweet.emoji for tweet in test_data]):
                        iteration_scorer.add(gold, prediction)

                    iteration_score = iteration_scorer.get_score()
                    iteration_macro_score = iteration_score.macro_f1
                    if config.epochs == 99999:
                        if iteration_macro_score > max_iteration_score:
                            max_iteration_score = iteration_macro_score
                            iterations_since_max = 0
                        else:
                            iterations_since_max += 1
                        if iterations_since_max >= config.max_non_improving_iterations:
                            stop = True

                    print("\n----- Iteration {} results for fold {} -----\n{}\n".format(iteration,
                                                                                        fold,
                                                                                        iteration_score),
                          file=log,
                          flush=True)
                    if config.confusion_matrix:
                        print("--- Matrix ---\n{}\n".format(iteration_scorer), file=log, flush=True)

                if config.checkpoint_saving:
                    model_path = os.path.join("models", MACHINE_NAME, "{}_{}_{}".format(config.id, fold, iteration))
                    if VERBOSE:
                        print("Saving model to '{}'...".format(model_path), file=log, flush=True)
                    model.save_model(model_path)
                    log_checkpoint(config.id, fold, iteration, model_path)
                if stop:
                    print("Stopping Training...", file=log, flush=True)
                    break
        else:
            model.train(train_data, epochs=config.epochs)

        # Score this fold.
        # Get a fold scorer/confusion matrix.
        if config.confusion_matrix:
            fold_scorer = ConfusionMatrix(class_count)
        else:
            fold_scorer = Scorer()

        predictions = model.batch_predict([tweet.text for tweet in test_data])
        for prediction, gold in zip(predictions, [tweet.emoji for tweet in test_data]):
            fold_scorer.add(gold, prediction)
            total_scorer.add(gold, prediction)

        # Print out the results
        print("\n----- Results for Fold {} -----\n{}\n".format(fold, fold_scorer.get_score()), file=log, flush=True)

        if VERBOSE:
            # Print out the result details
            print("--- Details ---", file=log, flush=True)
            print("Training data len: {} Testing data len: {}".format(len(train_data), len(test_data)), file=log, flush=True)
            print("Training data class counts: " + ", ".join([str(i) + ": " + str(count(train_data, i)) for i in range(class_count)]), file=log, flush=True)
            print("Testing  data class counts: " + ", ".join([str(i) + ": " + str(count(test_data, i)) for i in range(class_count)]), file=log, flush=True)

        # Print out the confusion matrix if enabled
        if config.confusion_matrix:
            print("\n--- Matrix ---\n{}\n".format(fold_scorer), file=log, flush=True)

    print("\n----- Results for all folds -----\n{}\n".format(total_scorer.get_score()), file=log, flush=True)
    if config.confusion_matrix:
        print("--- Matrix ---\n{}\n".format(total_scorer), file=log, flush=True)

    log.close()


def main(config_fn, recover=False):
    config = Config.from_json_obj(json.load(open(config_fn, "r")))
    print("Running with following config...\n{}".format(config))

    if config.model in NON_NN_MODELS:
        run_non_nn_model(config)
    else:
        run_nn_model(config, recover)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--recover", action="store_true")
    args = parser.parse_args()
    VERBOSE = args.verbose
    main(args.config[0], args.recover)
