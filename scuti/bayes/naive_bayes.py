"""
@author carterjfulcher

Naive Bayes Classifier is a conditional probability model. 

It can be defined as the probability p, given C, for n features:

p(C | x1, x2, ... xn) 

the posterior = (prior x likelihood) / evidence
"""

import pandas as pd 
import numpy as np

from scuti.utils.Distribution import Distribution 

class BayesClassifier(): 
	def __init__(self): 
		self._normalizing_constants = dict() 

	def _compute_noramlizing_constants(self, labels: np.array):
		for _class, count in list(zip(*np.unique(labels, return_counts=True))): 
			self._normalizing_constants[_class] = count / len(labels)
	
	def _feature_probability(self, _class, feature): #compute the probabilty of feature | class 
		pass

	def fit(self, features, labels): 
		self._compute_noramlizing_constants(labels)
		std, mean = Distribution.normal_distribution(pd.DataFrame(features, index=labels))


