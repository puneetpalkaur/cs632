# -*- coding: utf-8 -*-
"""

@author: Puneetpal Kaur

Assignment 1 Part 2: Spam filter

"""

import glob
import numpy as np
import pandas as pd
from tensorflow.contrib.keras.python.keras.preprocessing import text
from sklearn.preprocessing import LabelEncoder

import part1_classifier

input_file = input('Enter path for folder that contains csv files: ')
path_label_file = input('Enter path for label file: ')

# read csv files
filenames = glob.glob(input_file + "/0*.txt")
dfs = []
for filename in filenames:
    f = open(filename,'r')
    content = f.read()
    dfs.append(content)
    f.close()
# print(dfs)

dataframe = pd.DataFrame(data = dfs, columns=['body'])
df = pd.read_csv(path_label_file, header=None, delimiter=" ", names=['label','file_name'])
# print(df)
# axis - label index
merged_df = pd.concat([dataframe,df], axis =1)

# separate data

np.random.seed(0)
my_data = np.random.rand(len(merged_df))<0.5
train_data = merged_df[my_data]
test_data = merged_df[~my_data]

# assign data  train and test
x_train = train_data.body
y_train = train_data.label
x_test = test_data.body
y_test = test_data.label

# bag of words
max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(x_train) # only fit on train
x_train = tokenize.texts_to_matrix(x_train)
x_test = tokenize.texts_to_matrix(x_test)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(y_train)
train_y = encoder.transform(y_train)
test_y = encoder.transform(y_test)

# call custom classifier to train and test dataset
n_neighbors=3
myclassifier = part1_classifier.MyNearestNeighborClassifier(n_neighbors)

# fit dataset
myclassifier.fit(x_train,train_y)

# predict dataset
predict_method = myclassifier.predict(x_test,test_y)
