#!/usr/bin/env python

from keras.models import load_model
import tensorflow as tf

# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# import glob
# from PIL import Image
# from scipy.misc import imread, imresize

import rospy
import cv2 # OpenCV
from sensor_msgs.msg import CompressedImage, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Point, PoseStamped, Pose, PoseWithCovarianceStamped

class MLTag(object):
    # TODO: Add cmd_vel command based on where person supposedly is
    # TODO: Add logic if robot does not see person
    # TODO: Tag logic

    def __init__(self, model_name='convolutional_model_v2_half.h5'):
        rospy.init_node("ml_tag_node")

        self.my_model = load_model(model_name)
        self.my_graph = tf.get_default_graph()
        self.scan_ranges = []

        self.camera_subscriber = rospy.Subscriber("/camera/image_raw/compressed", CompressedImage, self.process_image)
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.process_scan)

        self.position_publisher = rospy.Publisher('/positions_pose_array', PoseArray, queue_size=10)
        self.position_pose_array = PoseArray()
        self.position_pose_array.header.frame_id = "base_link"

        self.object_publisher = rospy.Publisher('/object_marker', Marker, queue_size=10)
        self.my_object_marker = Marker()
        self.my_object_marker.header.frame_id = "base_link"
        self.my_object_marker.color.a = 0.5
        self.my_object_marker.color.g = 1.0
        self.my_object_marker.type = Marker.SPHERE
        self.my_object_marker.scale.x = 0.25
        self.my_object_marker.scale.y = 0.25
        self.my_object_marker.scale.z = 0.25

        self.model_object_publisher = rospy.Publisher('/model_object_marker', Marker, queue_size=10)
        self.my_model_object_marker = Marker()
        self.my_model_object_marker.header.frame_id = "base_link"
        self.my_model_object_marker.color.a = 0.5
        self.my_model_object_marker.color.b = 1.0
        self.my_model_object_marker.type = Marker.SPHERE
        self.my_model_object_marker.scale.x = 0.25
        self.my_model_object_marker.scale.y = 0.25
        self.my_model_object_marker.scale.z = 0.25


    def process_image(self, compressed_image_msg):
        print("Got image?")
        # Display compressed image:
        # http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(compressed_image_msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Show image
        cv2.imshow('cv_img', image_np)
        cv2.waitKey(2)

        # Resize image
        height, width = image_np.shape[:2]
        new_width = 200
        new_height = int(height * new_width * 1.0 / width)
        image_np_resized = cv2.resize(image_np, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        img_tensor = np.expand_dims(image_np_resized, axis=0) # Add 4th dimension it expects
        with self.my_graph.as_default():
            # Without using graph, it gives error: Tensor is not an element of this graph.
            # Could fix this by not doing image processing in the callback, and in the main run loop.
            # https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
            predicted = self.my_model.predict(img_tensor)
            # print("Predicted mouse: {}, shape: {}".format(predicted, predicted.shape))
            self.my_model_object_marker.pose.position.x = predicted[0]
            self.model_object_publisher.publish(self.my_model_object_marker)

    def process_scan(self, scan_msg):
        self.scan_ranges = scan_msg.ranges
        self.visualize_positions_in_scan()
        self.visualize_object_from_scan()

    def find_positions_in_scan(self):
        # Use front field of view of the robot's lidar to detect a person's x, y offset
        field_of_view = 35
        maximum_range = 3 # m

        # Cycle through ranges and filter out 0 or too far away measurements
        # Calculate the x, y coordinate of the point the lidar detected
        x_positions = []
        y_positions = []

        for angle in range(-1 * field_of_view, field_of_view):
            r = self.scan_ranges[angle]
            # print("angle: {}, r = {}".format(angle, r))
            if(r > 0 and r < maximum_range):
                # Convert angle to radians. Change negative angle to positive to output negative cos.
                # theta = math.radians(angle) if angle > 0 else math.radians(90 + angle)
                # Calculate the x, y coordinate
                theta = math.radians(angle)
                x_pos = r * math.cos(theta)
                y_pos = r * math.sin(theta)
                if(angle < 0):
                    x_pos != -1
                # print("x, y = {}, {} at angle = {}. theta = {}, r = {}".format(x_pos, y_pos, angle, math.degrees(theta), r))

                x_positions.append(x_pos)
                y_positions.append(y_pos)
        return (x_positions, y_positions)

    def find_object_from_scan(self):
        # Get the x, y coordinates of objects in the field of view
        x_positions, y_positions = self.find_positions_in_scan()

        if(len(x_positions) < 5 or len(x_positions) != len(y_positions)):
            # Not enough points
            return (-1, -1)

        center_of_mass = (sum(x_positions) * 1.0 / len(x_positions), sum(y_positions) * 1.0 / len(y_positions))

        return center_of_mass

    def visualize_positions_in_scan(self):
        x_positions, y_positions = self.find_positions_in_scan()
        all_poses = []

        for i in range(len(x_positions)):
            pose = Pose()
            pose.position.x = x_positions[i]
            pose.position.y = y_positions[i]
            all_poses.append(pose)

        self.position_pose_array.poses = all_poses
        self.position_publisher.publish(self.position_pose_array)

    def visualize_object_from_scan(self):
        x, y = self.find_object_from_scan()

        self.my_object_marker.header.stamp = rospy.Time.now()
        self.my_object_marker.pose.position.x = x
        self.my_object_marker.pose.position.y = y

        self.object_publisher.publish(self.my_object_marker)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    tag = MLTag()
    tag.run()
