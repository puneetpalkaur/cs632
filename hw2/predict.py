# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:57:09 2017

@author: Puneetpal Kaur
"""

""" This code demonstrates reading the test data and writing 
predictions to an output file.

It should be run from the command line, with one argument:

$ python predict_starter.py [test_file]

where test_file is a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.

(To test this script, you can use one of those).

This script will create an output file in the same directory 
where it's run, called "predictions.txt".

"""


import numpy as np
import sys

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

TEST_FILE = sys.argv[1]
MODEL_NAME = sys.argv[2]


from keras.models import load_model
model = load_model(MODEL_NAME)

data = np.load(TEST_FILE).item()

# these are images in exactly the same format
# as your train and validation set
images = data["images"]


# the testing data will also contains a unique id
# for each testing image
# because your training / validation doesn't have this
# we will generate some if this doesn't exist
if "ids" in data:
    print('ids are there ')
    ids = data["ids"]
else:
    # generate some random ids
    ids = list(range(0,len(images)))


# This file will be created if it does not exist
# and overwritten if it does
OUT_FILE = "predictions.txt"
BATCH_SIZE =20
predictions=model.predict(images,BATCH_SIZE,verbose=1)
   

# make a prediction on each image
# and write output to disk
out = open(OUT_FILE, "w")
for i, image in enumerate(images):
  image_id = ids[i]
  # this should be "1" for Cat and "0" for dog.
  pred = predictions[i]
  my_output = 1
  if pred[0] < 0.5:
      my_output = 0   
      
  line = str(image_id) + " " + str(my_output) + "\n"
  out.write(line)

out.close()
