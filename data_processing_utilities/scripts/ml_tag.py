#!/usr/bin/env python

from keras.models import load_model
import tensorflow as tf

# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import glob
# from PIL import Image
# from scipy.misc import imread, imresize

import rospy
import cv2 # OpenCV
from sensor_msgs.msg import CompressedImage, LaserScan

class MLTag(object):
    # TODO: Add prediction of where person should be based on laser_scan data (Add visualization)
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

    def process_image(self, compressed_image_msg):
        # Display compressed image:
        # http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(compressed_image_msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Show image
        # cv2.imshow('cv_img', image_np)
        # cv2.waitKey(2)

        # Resize image
        height, width = image_np.shape[:2]
        new_width = 200
        new_height = int(height * new_width * 1.0 / width)
        image_np_resized = cv2.resize(image_np, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        img_tensor = np.expand_dims(image_np_resized, axis=0) # Add 4th dimension it expects
        with self.my_graph.as_default():
            # Without using graph, it gives error: Tensor is not an element of this graph.
            # https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
            predicted = self.my_model.predict(img_tensor)
            print("Predicted mouse: {}".format(predicted[0]))

    def process_scan(self, scan_msg):
        self.scan_ranges = scan_msg.ranges

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    tag = MLTag()
    tag.run()
