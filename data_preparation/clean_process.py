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
    view_angle = int(70/2)     # only look at points in the forwardmost 70 degs
    infront = range(-view_angle, view_angle)
    # ranges[0:int(view_angle/2)]+ranges[int(360-view_angle/2):360]
    xs = []
    ys = []

    # loop through and grab points in desired view range
    for i in range(-view_angle, view_angle):
        if ranges[i] !=0:
            theta = math.radians(90+i)
            r = ranges[i]
            xf = math.cos(theta)*r
            yf = math.sin(theta)*r
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
        return(0, 0)   # TODO pick a return value for poor scans
    else:
        x_cord = sum(x)/len(x)
        y_cord = sum(y)/len(y)
        plt.plot(x, y, 'ro')
        plt.plot(0,0, 'bo', markersize=15)
        plt.plot(x_cord, y_cord, 'go', markersize=15)
        # plt.show()
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

if __name__ == '__main__':
    # location definitions
    data_path ='/home/anil/catkin_ws/src/comprobo18/robot_learning/data_processing_utilities/data/'
    folder_name = 'latest_person'
    path = data_path + folder_name + '/'
    metadata_csv = data_path + folder_name + '/' + 'metadata.csv'

    # image definitions
    os.chdir(path)
    filenames = glob.glob("*.jpg")

    # pull from metadata
    array_form = np.genfromtxt(metadata_csv, delimiter=",")
    lidar_all = array_form[:, 6:366]
    times = array_form[:,0]
    images = []
    object_xs = []
    object_ys = []

    unique = 0

    # loop through all images
    for i in range(lidar_all.shape[0]):
        print(times[i+1]-times[i])
        scan_now = lidar_all[i] # scan data for this index

        # process if scan isn't NaN (laser hasn't fired yet)
        if not np.isnan(scan_now[10]):
            points_x, points_y = process_scan(scan_now)
            xp, yp = center_of_mass(points_x, points_y)

            #  only add if CoM is defined, AKA object is in frame
            if xp != 0:
                if unique != xp:
                    print(i, xp, yp, math.degrees(math.atan2(xp, yp)))

                    unique = xp

                    # add image
                    img_name = filenames[i]
                    img_np = resize_image(img_name)
                    images.append(img_np)

                    # add object position
                    object_xs.append(xp)
                    object_ys.append(yp)

                    # verify
                    plt.imshow(img_np)
                    plt.show()



    adj_img = []
    adj_xs = []
    adj_ys = []

    # save all data
    save_path = data_path + folder_name + '_data' '.npz'
    np.savez_compressed(save_path, imgs=images, object_x=object_xs, object_y=object_ys)
