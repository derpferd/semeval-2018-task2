import os

from run import get_class_count, get_test_train_sets, load_tweets
from hopper.confusion_matrix import ConfusionMatrixWithExamples
from hopper.model_char_lstm import CharLSTMCNNModel


model_name = "char_cnn_lstm_es_train_128_128_0_13"
lang = "es"

model = CharLSTMCNNModel(80, 128, 128)
model.load_model(os.path.join("models", model_name))

tweets = load_tweets("data/train/{}_train".format(lang))
class_count = get_class_count(tweets)
scorer = ConfusionMatrixWithExamples(class_count)

train_data, test_data = get_test_train_sets(tweets, 0, 10)
predictions = model.batch_predict([tweet.text for tweet in test_data])

for prediction, gold, text in zip(predictions, [tweet.emoji for tweet in test_data], [tweet.text for tweet in test_data]):
    scorer.add(gold, prediction, text, model.tokenize(text))

scorer.dump_json("{}.{}.matrix.json".format(model_name, lang))