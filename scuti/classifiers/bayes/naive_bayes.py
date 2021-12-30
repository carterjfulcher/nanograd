"""
@author carterjfulcher

Naive Bayes Classifier is a conditional probability model. 

It can be defined as the probability p, given C, for n features:

p(C | x1, x2, ... xn) 

the posterior = (prior x likelihood) / evidence
"""

import pandas as pd 
import numpy as np

from scuti.utils.distribution import Distribution 

class BayesClassifier(): 
	def __init__(self): 
		self._normalizing_constants = dict() 
		
		#matrices of shape (n_classes, n_features)
		self._variances = [] 
		self._means = [] 

	def _compute_noramlizing_constants(self, labels: np.array):
		for _class, count in list(zip(*np.unique(labels, return_counts=True))): 
			self._normalizing_constants[_class] = count / len(labels)
	
	def _feature_probability(self, _class, feature): #compute the probabilty of feature | class 
		pass

	def fit(self, features, labels): 
		#compute the mean and variances of each feature
		self._compute_noramlizing_constants(labels)
		for _class, _ in self._normalizing_constants.items():
			class_filtered = [item for index, item in enumerate(features) if labels[index] == _class]
			class_means, class_variance = [], [] 
			
			for feature_index in range(features.shape[1]):
				feature_set = np.array([i[feature_index] for i in class_filtered])
				class_means.append(feature_set.mean())
				class_variance.append(sum([(x - np.mean(feature_set))**2 for x in feature_set])/(len(feature_set)-1))

			self._means.append(class_means) 
			self._variances.append(class_variance)

		self._variances = np.array(self._variances)
		self._means = np.array(self._means)
			