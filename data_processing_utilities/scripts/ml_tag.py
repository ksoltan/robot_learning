#!#!/usr/bin/env python
from keras.models import load_model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
from scipy.misc import imread, imresize

# Load Model
model = load_model('convolutional_model_v1.h5')

# Testing on already collected data:
folder_name = 'amy_follow'
# Katya
data_path = '/home/ksoltan/catkin_ws/src/robot_learning/data_processing_utilities/data/'
path = data_path + folder_name

os.chdir(path)
filenames = glob.glob("*.jpg")

# Get size of images
filename = filenames[0]
sample_img = Image.open(filename, 'r')
print("height: {}, width: {}, aspect: {}".format(sample_img.height, sample_img.width, 1.0 * sample_img.height/sample_img.width))
aspect = 1.0 * sample_img.height / sample_img.width
width = 200
height = int(width*aspect)
new_size = width, height

# Select index of image you would like to predict mouse_x.
index = 100
images = []
img = Image.open(filenames[index], 'r')
resize = img.resize(new_size)
array = np.array(resize)
images.append(array)

predicted = model.predict(np.array(images))
plt.imshow(images[0])
print("Predicted mouse: {}".format(predicted[0]))
