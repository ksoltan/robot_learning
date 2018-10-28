import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
from scipy.misc import imread, imresize


path = '/home/anil/catkin_ws/src/comprobo18/robot_learning/data_preparation/test1'

foldername = 'test1'
dataname = 'metadata.csv'

os.chdir(path)
filenames = glob.glob("*.jpg")

filename = '/home/anil/catkin_ws/src/comprobo18/robot_learning/data_preparation/test1/0000002101.jpg'
sample_img = Image.open(filename, 'r')
aspect = sample_img.height/sample_img.width
width = 200
height = int(width*aspect)
new_size = width, height

m_dim1 = height
m_dim2 = width
m_dim3 = 3 # rgb
m_dim4 = len(filenames) # num pics

images = np.empty((m_dim1, m_dim2, m_dim3, m_dim4))

img = Image.open(filename, 'r')
resized = img.resize(new_size)
data = np.array(resized)

# plt.imshow(data)
# plt.show()
index = 0

for name in filenames:
    img = Image.open(name, 'r')
    resize = img.resize(new_size)
    array = np.array(resize)
    images[:,:,:,index] = array
    index += 1
