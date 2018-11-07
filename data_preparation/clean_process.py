# Given a folder of images and a metadata.csv file, output an npz file with an imgs, spatial x, and spatial x dimensions.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import math
from PIL import Image
from scipy.misc import imread, imresize

def process_scan(ranges):
    """
    process a 360 point set of laser data in a certain viewing range.

    inputs: list of ranges from the laser scan
    output: lists of x and y points within viewing angle and range
    """
    max_r = 1.5
    view_angle = int(70 / 2)     # only look at points in the forwardmost 70 degs
    infront = range(-view_angle, view_angle)
    # ranges[0:int(view_angle/2)]+ranges[int(360-view_angle/2):360]
    xs = []
    ys = []

    # loop through and grab points in desired view range
    for i in range(-view_angle, view_angle):
        if ranges[i] != 0:
            theta = math.radians(90 + i)
            r = ranges[i]
            xf = r * math.cos(theta)
            yf = r * math.sin(theta)
            xs.append(xf)
            ys.append(yf)

    return(xs, ys)

def center_of_mass(x, y):
    """
    compute the center of mass in a lidar scan.

    inputs: x and y lists of cleaned laser data
    output: spatial x and y coordinate of the CoM
    """
    if len(x) < 4:     # if below a threshold of grouped points
        return(np.inf, np.inf)
    else:
        x_cord = sum(x)/len(x)
        y_cord = sum(y)/len(y)

        plt.plot(x, y, 'ro')
        plt.plot(0,0, 'bo', markersize=15)
        plt.plot(x_cord, y_cord, 'go', markersize=15)
        plt.ylim(-2,2)
        plt.xlim(-2,2)        # plt.show()
        return (x_cord, y_cord)

def resize_image(img_name):
    """
    load and resize images for the final numpy array.

    inputs: filename of an image
    output: resized image as a numpy array
    """
    # new size definition
    width = 200
    height = 150
    new_size = width, height

    img = Image.open(img_name, 'r')
    resize = img.resize(new_size)
    array = np.array(resize)
    return array

def find_corresponding_scan(image_time, scan_times, start_idx):
    max_tolerance = 0.015
    while start_idx < len(scan_times):
        diff = abs(scan_times[start_idx] - image_time)
        # print("Idx: {}, Diff: {}".format(start_idx, abs(scan_times[start_idx] - image_time)))
        if diff < max_tolerance:
            return (start_idx, diff)
        start_idx += 1
    return None


if __name__ == '__main__':
    # location definitions
    # # Katya
    data_path = '/home/ksoltan/catkin_ws/src/robot_learning/data_processing_utilities/data/'
    # Anil
    # data_path ='/home/anil/catkin_ws/src/comprobo18/robot_learning/data_processing_utilities/data/'
    folder_name = 'anil_shining_2'
    # folder_name = 'latest_person'

    path = data_path + folder_name + '/'
    metadata_csv = data_path + folder_name + '/' + 'metadata.csv'

    # image definitions
    os.chdir(path)
    filenames = glob.glob("*.jpg")

    # pull from metadata
    array_form = np.genfromtxt(metadata_csv, delimiter=",")
    lidar_all = array_form[:,6:366]
    pic_times = array_form[:,0]
    lidar_times = array_form[:,-1]

    images = []
    object_xs = []
    object_ys = []

    i_s = []
    j_s = []
    # loop through all images
    for i in range(lidar_all.shape[0]-26):
        for j in range(i,i+25):
            delta = lidar_times[j]-pic_times[i]
            if abs(delta) < 0.025:
                i_s.append(i)
                j_s.append(j)

                # print('pic', i)
                # print('lid', j)
                # print('delta', delta)
                # print('------------------')
                break

    imgs_a = []
    xs_a = []
    ys_a = []

    for i in range(len(i_s)):
        img_ind = i_s[i]
        lid_ind = j_s[i]

        scan_now = lidar_all[lid_ind] # scan data for this index

        # process if scan isn't NaN (laser hasn't fired yet)
        if not np.isnan(scan_now[10]):
            points_x, points_y = process_scan(scan_now)
            xp, yp = center_of_mass(points_x, points_y)

            #  only add if CoM is defined, AKA object is in frame
            if xp != np.inf:
                # print(pic_times[img_ind]-lidar_times[lid_ind], xp, yp, round(math.degrees(math.atan2(xp, yp)),2))

                # add image
                img_name = filenames[img_ind]
                img_np = resize_image(img_name)
                imgs_a.append(img_np)

                # add object position
                xs_a.append(xp)
                ys_a.append(yp)

                # verify
                # plt.show()

                plt.imshow(img_np)
                # plt.show()

    print(len(imgs_a))
    # save all data
    save_path = data_path + folder_name + '_data' '.npz'
    np.savez_compressed(save_path, imgs=imgs_a, object_x=xs_a, object_y=ys_a)
