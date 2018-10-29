# Given a folder of images and a metadata.csv file, output an npz file with an imgs, mouse_x, and mouse_y columns.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
from scipy.misc import imread, imresize

folder_name = 'test2'
# Katya
data_path = '/home/ksoltan/catkin_ws/src/robot_learning/data_processing_utilities/data/'
# # Anil
# data_path ='/home/anil/catkin_ws/src/comprobo18/robot_learning/data_preparation/'

path = data_path + folder_name

metadata_name = 'metadata.csv'

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

# plt.imshow(data)
# plt.show()
index = 0
images = []
# Create numpy array of resized images
for name in filenames:
    img = Image.open(name, 'r')
    resize = img.resize(new_size)
    array = np.array(resize)
    # images[:,:,:,index] = array
    images.append(array)
    index += 1

# Create numpy array of all x and y mouse positions
METADATA_CSV = data_path + folder_name + '/' + metadata_name
df = pd.read_csv(METADATA_CSV, ',')[['image_file_name', 'mouse_x', 'mouse_y']]
print(df.head())
print(df.info())

SAVE_FILENAME = data_path + folder_name + '.npz'
np.savez_compressed(SAVE_FILENAME, imgs=images, mouse_x=df.mouse_x, mouse_y=df.mouse_y)
test_data = np.load(SAVE_FILENAME)
print(test_data['mouse_x'].shape)
