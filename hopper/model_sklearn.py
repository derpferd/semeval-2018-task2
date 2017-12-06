# Author: Jonathan Beaulieu
from sklearn.externals import joblib

from hopper import Model


class SKLearnModel(Model):
	""" Just a template class for Model which use SKLearn.
		This is so we don't have to duplicate the save and model methods, which are the same for all SKLearn Models.
	"""
    def save_model(self, path):
        joblib.dump(self.classif, path)

    def load_model(self, path):
        self.classif = joblib.load(path)
