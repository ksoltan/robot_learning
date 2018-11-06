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
        # plt.plot(x, y, 'ro')
        # plt.plot(0,0, 'bo', markersize=15)
        # plt.plot(x_cord, y_cord, 'go', markersize=15)
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
     # Anil
    # data_path = '/home/anil/catkin_ws/src/comprobo18/robot_learning/data_processing_utilities/data/'
    # Katya
    data_path = '/home/ksoltan/catkin_ws/src/robot_learning/data_processing_utilities/data/'
    folder_name = 'anil_shining'
    path = data_path + folder_name + '/'
    metadata_csv = data_path + folder_name + '/' + 'metadata.csv'

    # image definitions
    os.chdir(path)
    filenames = glob.glob("*.jpg")

    # pull from metadata
    array_form = np.genfromtxt(metadata_csv, delimiter=",")
    lidar_all = array_form[:, 6:366]

    df = pd.read_csv(metadata_csv, ',')[['stamp', 'object_from_scan_x', 'object_from_scan_y', 'object_from_scan_stamp', 'lidar_stamp']]
    ml_tag_object_xs = []
    ml_tag_object_ys = []
    ml_tag_object_times = []

    # initialize key data lists
    times = array_form[:,0]

    images = []
    object_xs = []
    object_ys = []

    unique = 0
    all_diffs = []

    for i in range(lidar_all.shape[0] - 1):
        # print("{}/{}...".format(i, lidar_all.shape[0]))
        image_time = df.stamp[i]
        # res = find_corresponding_scan(image_time, df.lidar_stamp, i)
        res = find_corresponding_scan(image_time, df.object_from_scan_stamp, i)
        if(res != None):
            scan_now_idx = res[0]
            diff = res[1]
            print("Found corresponding times: img: {}  >>> scan: {}".format(image_time, df.lidar_stamp[scan_now_idx]))
            unique += 1
            all_diffs.append(diff)
            print("scan_now_idx: {}, img_idx: {}".format(scan_now_idx, i))
            # scan_now = lidar_all[scan_now_idx]
            # points_x, points_y = process_scan(scan_now)
            # xp, yp = center_of_mass(points_x, points_y)
            xp, yp = df.object_from_scan_x[scan_now_idx], df.object_from_scan_y[scan_now_idx]
            print("xp: {}, yp: {}".format(xp, yp))
            # If we have a center of mass
            if(xp != np.inf):
                print(i, xp, yp, math.degrees(math.atan2(xp, yp)))
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

                # ml_tag_object_times.append(abs(image_time - df.object_from_scan_stamp[i]))
                # ml_tag_object_xs.append(df.object_from_scan_x[i])
                # ml_tag_object_ys.append(df.object_from_scan_y[i])

    print("Found {} corresponding scan - lidar".format(unique))
    # # loop through all imagestest_data_faster
    # for i in range(lidar_all.shape[0]):
    #     # print("{}/{}...".format(i, lidar_all.shape[0]))
    #     # print(times[i+1]-times[i])
    #     scan_now = lidar_all[i] # scan data for this index
    #
    #     # process if scan isn't NaN (laser hasn't fired yet)
    #     if not np.isnan(scan_now[10]):
    #         points_x, points_y = process_scan(scan_now)
    #         xp, yp = center_of_mass(points_x, points_y)
    #
    #         #  only add if CoM is defined, AKA object is in frame
    #         if xp != 0:
    #             if unique != xp:
    #                 print(i, xp, yp, math.degrees(math.atan2(xp, yp)))
    #
    #                 unique = xp
    #
    #             time_scan = df.lidar_stamp[i]
    #             time_img = df.stamp[i]
    #             print("Time diff: {}".format(time_scan - time_img))
    #
    #             # Use only images taken around the same time lidar is published.
    #             if(abs(time_scan - time_img) < 0.5):
    #                 # Use this value.
    #                 # print(i, xp, yp, math.degrees(math.atan2(xp, yp)))
    #
    #                 # add image
    #                 img_name = filenames[i]
    #                 img_np = resize_image(img_name)
    #                 images.append(img_np)
    #
    #                 # add object position
    #                 object_xs.append(xp)
    #                 object_ys.append(yp)
    #
    #                 # verify
    #                 plt.imshow(img_np)
    #                 # plt.show()
    #
    #                 # Check the time stamp of the ml_tag prediction.
    #                 ml_tag_object_times.append(time_scan - df.object_from_scan_stamp[i])
    #                 ml_tag_object_xs.append(df.object_from_scan_x[i])
    #                 ml_tag_object_ys.append(df.object_from_scan_y[i])
    #
    #                 plt.show()

    # save all data
    # save_path = data_path + folder_name + '_data' '.npz'
    # np.savez_compressed(save_path, imgs=images, object_x=object_xs, object_y=object_ys, ml_tag_object_x=ml_tag_object_xs, ml_tag_object_y=ml_tag_object_ys, ml_tag_object_stamp_diff=ml_tag_object_times)
