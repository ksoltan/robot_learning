# Given a folder of images and a metadata.csv file, output an npz file with an imgs, mouse_x, and mouse_y columns.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
from scipy.misc import imread, imresize

folder_name = 'ball_dataset_classroom'

# Katya
data_path = '/home/ksoltan/catkin_ws/src/robot_learning/data_processing_utilities/data/'
# Anil
# data_path ='/home/anil/catkin_ws/src/comprobo18/robot_learning/data_processing_utilities/data/'

path = data_path + folder_name + '/'

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


# Create numpy array of all x and y mouse positions
METADATA_CSV = data_path + folder_name + '/' + metadata_name
df = pd.read_csv(METADATA_CSV, ',')[['image_file_name', 'object_from_scan_x', 'object_from_scan_y']]
print(df.head())
print(df.info())

images = []
object_xs = []
object_ys = []
# Loop through lidar predicted object positions and save only those that do not contain 0, 0
for index in range(len(df.object_from_scan_x)):
    x = df.object_from_scan_x[index]
    y = df.object_from_scan_y[index]
    if(x == 0.0 and y == 0.0):
        continue

    # Add image
    img_name = filenames[index]
    img = Image.open(img_name, 'r')
    resize = img.resize(new_size)
    array = np.array(resize)
    images.append(array)

    # Add object position
    object_xs.append(x)
    object_ys.append(y)


#
# # plt.imshow(data)
# # plt.show()
# index = 0
# images = []
# # Create numpy array of resized images
# for name in filenames:
#     img = Image.open(name, 'r')
#     resize = img.resize(new_size)
#     array = np.array(resize)
#     # images[:,:,:,index] = array
#     images.append(array)
#     index += 1


SAVE_FILENAME = data_path + folder_name + '_data' '.npz'
np.savez_compressed(SAVE_FILENAME, imgs=images, object_x=object_xs, object_y=object_ys)
test_data = np.load(SAVE_FILENAME)
print(test_data['object_x'].shape)
