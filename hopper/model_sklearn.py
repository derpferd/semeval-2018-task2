from sklearn.externals import joblib

from hopper import Model


class SKLearnModel(Model):
    def save_model(self, path):
        joblib.dump(self.classif, path)

    def load_model(self, path):
        self.classif = joblib.load(path)
