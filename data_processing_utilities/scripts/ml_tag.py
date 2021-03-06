#!/usr/bin/env python

from keras.models import load_model
import tensorflow as tensorflow

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
from neato_node.msg import Bump
import tf

# from data_processing_utilities.msgs import ImageScanStamped

class MLTag(object):
    # TODO: Add cmd_vel command based on where person supposedly is
    # TODO: Add logic if robot does not see person
    # TODO: Tag logic

    def __init__(self, model_name='convolutional_model_v5.h5'):
        rospy.init_node("ml_tag_node")

        self.my_model = load_model(model_name)
        self.my_graph = tensorflow.get_default_graph()
        self.scan_ranges = []
        self.is_tagger = True # Switch state based on whether robot is tagging or running away
        self.got_scan = False
        self.ready_to_process = False

        self.camera_subscriber = rospy.Subscriber("/camera/image_raw/compressed", CompressedImage, self.process_image)
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.process_scan)
        self.bump_subscriber = rospy.Subscriber("/bump", Bump, self.process_bump)

        # Publisher for logging
        self.object_from_scan_publisher = rospy.Publisher("/object_from_scan", PoseStamped, queue_size=10)

        # Transform
        self.tf_listener = tf.TransformListener()

        # Visuzaliations
        self.position_publisher = rospy.Publisher('/positions_pose_array', PoseArray, queue_size=10)
        self.position_pose_array = PoseArray()
        self.position_pose_array.header.frame_id = "base_link"

        # self.image_scan_publisher = rospy.Publisher('/image_scan_pose', ImageScanStamped, queue_size=10)
        # self.last_scan_msg = None
        # self.last_image_msg = None

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
        # Display compressed image:
        # http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
        #### direct conversion to CV2 ####
        # if(self.got_scan and not self.ready_to_process):
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
            # print("Model predict: x: {}, y:{}, theta: {}".format(predicted[0][0], predicted[0][1], math.degrees(math.atan2(predicted[0][0], predicted[0][1]))))
            self.my_model_object_marker.pose.position.x = predicted[0][0]
            self.my_model_object_marker.pose.position.y = predicted[0][1]
            self.model_object_publisher.publish(self.my_model_object_marker)
            # self.last_image_msg = compressed_image_msg
            # self.got_scan = False
            # self.ready_to_process = True

    def process_scan(self, scan_msg):
        self.scan_ranges = scan_msg.ranges
        self.visualize_positions_in_scan()
        self.visualize_object_from_scan()
        # if(not self.ready_to_process):
        #     self.scan_ranges = scan_msg.ranges
        #     self.last_scan_msg = scan_msg
        #     self.got_scan = True

    def process_bump(self, bump_msg):
        pass

    def find_poses_in_scan(self):
        # Use front field of view of the robot's lidar to detect a person's x, y offset
        field_of_view = 40
        maximum_range = 2 # m

        # Cycle through ranges and filter out 0 or too far away measurements
        # Calculate the x, y coordinate of the point the lidar detected
        poses = []

        for angle in range(-1 * field_of_view, field_of_view):
            r = self.scan_ranges[angle]
            # print("angle: {}, r = {}".format(angle, r))
            if(r > 0 and r < maximum_range):
                try:
                    # Confirm that transform exists.
                    (trans,rot) = self.tf_listener.lookupTransform('/base_link', '/base_laser_link', rospy.Time(0))
                    # Convert angle to radians. Adjust it to compensate for lidar placement.
                    theta = math.radians(angle + 180)
                    x_pos = r * math.cos(theta)
                    y_pos = r * math.sin(theta)

                    # Use transform for correct positioning in the x, y plane.
                    p = PoseStamped()
                    p.header.stamp = rospy.Time.now()
                    p.header.frame_id = 'base_laser_link'

                    p.pose.position.x = x_pos
                    p.pose.position.y = y_pos

                    p_model = PoseStamped()
                    p_model.header.stamp = rospy.Time.now()
                    p_model.header.frame_id = 'base_laser_link'

                    p_model.pose.position.x = self.my_model_object_marker.pose.position.x
                    p_model.pose.position.y = self.my_model_object_marker.pose.position.y


                    p_base_link = self.tf_listener.transformPose('base_link', p)
                    p_model_base_link = self.tf_listener.transformPose('base_link', p_model)
                    # print("{}, {} at angle {}".format(p_base_link.pose.position.x, p_base_link.pose.position.y, math.degrees(theta)))
                    print("Lidar predict: x: {}, y:{}, theta: {}".format(p_base_link.pose.position.x, p_base_link.pose.position.y, math.degrees(theta)))
                    print("Lidar predict: x: {}, y:{}, theta: {}".format(p_model_base_link.pose.position.x, p_model_base_link.pose.position.y, math.degrees(theta)))

                    # Only care about the pose
                    poses.append(p_base_link.pose)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
        # Return a list of poses (no header)
        return poses

    def find_object_from_scan(self):
        # Get the x, y coordinates of objects in the field of view
        poses = self.find_poses_in_scan()
        min_points_for_object = 3

        if(len(poses) < min_points_for_object):
            # Not enough points
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = "base_link"
            self.object_from_scan_publisher.publish(pose_stamped)
            return (0, 0)

        # Not the most efficient list traversal (double), but we don't have that many values.
        center_of_mass = (sum([pose.position.x for pose in poses]) * 1.0 / len(poses),
                            sum([pose.position.y for pose in poses]) * 1.0 / len(poses))

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose.position.x = center_of_mass[0]
        pose_stamped.pose.position.y = center_of_mass[1]
        self.object_from_scan_publisher.publish(pose_stamped)

        return center_of_mass

    def visualize_positions_in_scan(self):
        poses = self.find_poses_in_scan()

        self.position_pose_array.poses = poses
        self.position_publisher.publish(self.position_pose_array)

    def visualize_object_from_scan(self):
        x, y = self.find_object_from_scan()

        self.my_object_marker.header.stamp = rospy.Time.now()
        self.my_object_marker.pose.position.x = x
        self.my_object_marker.pose.position.y = y

        self.object_publisher.publish(self.my_object_marker)

    def run(self):
        # while not rospy.is_shutdown():
        #     if(self.ready_to_process):
        #         self.visualize_positions_in_scan()
        #         # Publish an image/scan msg
        #         self.publish_image_scan()
        rospy.spin()

    # def publish_image_scan(self):
    #     msg = ImageScanStamped()
    #     msg.header.stamp = rospy.Time.now()
    #     msg.image = self.last_image_msg
    #     msg.scan = self.last_scan_msg
    #     x, y = self.visualize_object_from_scan()
    #     msg.pose.position.x = x
    #     msg.pose.position.y = y
    #     self.image_scan_publisher.publish(msg)
    #     self.ready_to_process = False

if __name__ == "__main__":
    tag = MLTag()
    tag.run()
