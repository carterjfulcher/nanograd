"""
@author carterjfulcher

Naive Bayes Classifier is a conditional probability model. 

It can be defined as the probability p, given C, for n features:

p(C | x1, x2, ... xn) 

the posterior = (prior x likelihood) / evidence
"""

import pandas as pd 
import numpy as np
import math

from scuti.utils.distribution import Distribution 

class BayesClassifier(): 
	def __init__(self): 
		self._normalizing_constants = dict() 
		
		#matrices of shape (n_classes, n_features)
		self._variances = [] 
		self._means = [] 

		#parameters of normal distribution
		self._mean = 0
		self._std = 0

	def _compute_noramlizing_constants(self, labels: np.array):
		for _class, count in list(zip(*np.unique(labels, return_counts=True))): 
			self._normalizing_constants[_class] = count / len(labels)
	
	def _feature_probability(self, _class, x): #compute the probabilty of feature | class 
		return 1 / math.sqrt(2 * math.pi * self._std ** 2) * math.exp(-(x - self._mean) ** 2 / (2 * self._std ** 2))

	def fit(self, features, labels): 
		#compute the mean and variances of each feature
		self._mean, self._std = Distribution.normal_distribution(features)
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

	def forward(self, feature_set: np.ndarray, ): #compute the posterior 
		assert isinstance(self._variances, np.ndarray), 'Error: Model has not been fit. Please use fit() method'

		#take the argmax of the posteriors
		posteriors = {k: [] for k in self._normalizing_constants.keys()}
		for _class in posteriors.keys():
			for feature in feature_set: 
				posteriors[_class].append(
					self._feature_probability(_class, feature)
				)
		posteriors = {k: np.prod(j) for k, j in posteriors.items()} #take the product 
		return max(posteriors, key=posteriors.get)
			