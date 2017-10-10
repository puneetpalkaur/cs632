# -*- coding: utf-8 -*-
"""
@author: Puneetpal Kaur

Purpose: Train model on the Iris training set
and print out accuracy on the test set

"""
import numpy as np
from sklearn import datasets
# Import custom classifier class
import part1_classifier
# load iris dataset
iris = datasets.load_iris()
# observed data 
x = iris.data
# external variable
y = iris.target
np.unique(y)
# split data into training and test data
# seed the random generator
np.random.seed(0)
# Randomly permute a sequence
indices = np.random.permutation(len(x))
x_train = x[indices[:-10]]
y_train = y[indices[:-10]]
x_test = x[indices[-10:]]
y_test = y[indices[-10:]]
# call custom classifier to train and test dataset
n_neighbors=3
myclassifier = part1_classifier.MyNearestNeighborClassifier(n_neighbors)
# fit dataset
myclassifier.fit(x_train,y_train)
# predict dataset
predict_method = myclassifier.predict(x_test,y_test)
