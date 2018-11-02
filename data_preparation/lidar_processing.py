#!/usr/bin/env python
"""quick script for trying to pull spatial x, y from metadata"""

from __future__ import print_function
from geometry_msgs.msg import PointStamped, PointStamped, Twist
from std_msgs.msg import Header
from neato_node.msg import Bump
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import time, numpy, math, rospy, statistics

def process_scan(ranges):
    """ process a 360 point set of laser data in a certain viewing range """
    max_r = 1.5
    view_angle = 80     # only look at points in the forwardmost 70 degs
    infront = ranges[0:int(view_angle/2)]+ranges[int(360-view_angle/2):360]

    xs = []
    ys = []

    # loop through and grab points in desired view range
    for i in range(len(ranges)):
        if i<len(infront):
            if infront[i] !=0 and infront[i] < max_r:
                if i >= view_angle/2:
                    theta = math.radians(90-(view_angle-i))
                else:
                    theta = math.radians(i+90)
                r = infront[i]
                xf = math.cos(theta)*r
                yf = math.sin(theta)*r
                xs.append(xf)
                ys.append(yf)

    return(xs, ys)

def center_of_mass(x, y):
    """ with arguments as lists of x and y values, compute center of mass """
    if len(x) < 4:          # if below a threshold of grouped points
        return(0, 0)   # TODO pick a return value for poor scans

    x_cord = sum(x)/len(x)
    y_cord = sum(y)/len(y)
    plt.plot(x_cord, y_cord, 'go', markersize=15)
    return (x_cord, y_cord)

if __name__ == '__main__':
    path = '/home/anil/catkin_ws/src/comprobo18/robot_learning/data_processing_utilities/data/'
    folder = 'mydataset'
    look_in = path+folder + '/'   # final path for metadata

    filename = 'metadata.csv'
    file_csv = look_in + filename

    array_form = numpy.genfromtxt(file_csv, delimiter=",")
    lidar_all = array_form[:, 6:366]
    lidar_label = []

    ind = 0
    for i in range(lidar_all.shape[0]):
        scan_now = lidar_all[i,:]

        if not numpy.isnan(scan_now[10]):
            points_x, points_y = process_scan(scan_now)
            xp, yp = center_of_mass(points_x, points_y)

            if xp != 0:
                # lidar_label[ind,0] = i
                # lidar_label[ind,1] = xp
                # lidar_label[ind,2] = yp
                # ind += 1
                lidar_label.append([i, xp, yp])
                print(ind, i, xp, yp, math.degrees(math.atan2(xp, yp)))

                # plt.plot(points_x, points_y, 'ro')
                # plt.plot(0,0, 'bo', markersize=15)
                # plt.show()

    lidar_label = numpy.array(lidar_label)
    print(lidar_label[:,0])
    SAVE_FILENAME = path + folder + '.npz'
    numpy.savez_compressed(SAVE_FILENAME, indices=lidar_label[:,0], xs=lidar_label[:,1], ys=lidar_label[:,2])
    """
    # loop through images and get spatial x and y
    for i in range(lidar_all.shape[0]):
        lidar_here = lidar_all[i,:]
        xs, ys = process_scan(lidar_here)
        xp, yp = center_of_mass(xs, ys)

        lidar_label[i,0] = xp
        lidar_label[i,1] = yp
        print(xp, yp)
    """
