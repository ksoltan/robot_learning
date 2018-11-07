# Robot Learning: Person Detection
## Project Goal
The goal of our project was to use the Neato robot's camera to interact with a human and play the game of tag. The game starts with a state machine that qualifies the Neato as either “it” or “not it.” The mode dictates whether or not the Neato is actively chasing the person in its view or driving away. Switching between the modes is toggled with the bump sensor. We focused on the processing of camera data to locate a human in the image, outputting a position that the Neato would chase or avoid. The camera is first used to gather a training set of images to build a convolutional neural network for mapping a new image to a person’s position. This trained model would then be downloaded and used live on the Neato for locating the person and adjusting its linear and angular velocities appropriately. Due to challenges in collecting good data, our resulting model was not quite successful in calculating the position of the human in the image.

# Approach
To identify the position of a person relative to the robot from an image, we trained a multi-layered neural network with four convolution/max-pool layers and a final flatten/dense layer to output an x and y coordinate. We collected a data set of images taken from the Neato's camera and the approximate center of mass of the person based on an average of laser scan points in the frame of view of the camera. Because the image and lidar sampling happened at different rates, before training, each image was paired with a scan result with a maximum timestamp difference of 0.025s. Additionally, the images were scaled down to 200 x 150 pixels instead of 680 x 480. You can find our training code [here](https://colab.research.google.com/drive/1UaE06H4dS8kt_A7o_D8_NWij7EhDyHtn).

![](https://github.com/ksoltan/robot_learning/blob/master/documentation/data_record_ml_tag_video.gif)
*Data collection included sampling camera images from the Neato. The ground truth position of the person was taken as the average position of the laser scan points in the field of view of the camera which were less than 2 m away from the robot. Above, the red points are laser scan points, red arrows are the points contributing to person detection, and the green sphere represents the estimate of the person's position.*

The resulting model was unfortunately not very accurate, even on its own training data. There was an extremely high variability in the predicted person positions, as seen in the graphs below:

Training Data:
![](https://github.com/ksoltan/robot_learning/blob/master/documentation/predicted_x_train.png)
![](https://github.com/ksoltan/robot_learning/blob/master/documentation/predicted_y_train.png)

Test Data:
![](https://github.com/ksoltan/robot_learning/blob/master/documentation/predicted_x_test.png)
![](https://github.com/ksoltan/robot_learning/blob/master/documentation/predicted_y_test.png)

The saliency map shows that the neural network was not identifying the key locations of the person in the image, explaining the model's discrepancy. The points of interest (more cyan) are in the correct general area, but are extremely dispersed:
![](https://github.com/ksoltan/robot_learning/blob/master/documentation/saliency_many.png)

## Design Decisions
One design decision we made was using the lidar to classify the position of the person instead of using a mouse to indicate the location of the person in the image. The mouse approach did not allow us to account for how the person's depth changes in the image as they move around the space. Additionally, outputting a position in the room's spatial coordinates had the additional advantage of a simpler conversion into angular and linear velocity commands to move the robot.

Another reason for using the lidar to label our training set was the goal of automating and parallelizing our image recording and classification. Both the data collection and labelling was done with ROS topics running on neato-hardware, which in an ideal world would make accurate data collection in large batches a fast and simple process.

## Challenges
- We started by training the model to identify a person in the image’s 2D space. However, bad mouse-tracking data and a generally poor data set for this type of relationship led to frustrating results that never made any valuable predictions.
- We tried a couple different approaches to data logging and classifying the center of mass. This should have been much more simple but was not porting over from the warmup project as successfully as we would have hoped and led to delays.
- We pivoted to focus 100% on the lidar-based classification and ran into many challenges related to image classification. A critical error was the “interpolated_scan” topic being unable to function correctly, creating the subsequent challenge of correlating lidar data and image data that was recorded at different frequencies.

## Future Improvements
- Experiment with different classification methods and tuning our dataset to be as clear and correlative to our labels as possible. Although we used a classic convolutional neural network approach, there may have been other parameters or paths to explore which could have worked better for our data.
- If our model was functional, we would have dove further into the robot implementation. There are many design decisions and levels of depth we could have taken the on-robot implementation to. Having time to experiment with and develop this part of the project would have been another exciting challenge to handle.

## Lessons Learned
- Verifying and understanding large datasets. Building algorithms for both robust data collection, processing, and verification at the very beginning of the project would have saved us a ton of time at the end when we needed time to tune our neural net or rapidly take sets of data.
- Being very deliberate with our decision making around the “black box” that is top-down machine learning. We definitely had a hard time locating failure modes between our different datasets, processing methods, and the neural net itself. Having a better understanding of how each of these pieces affect the performance of the model and what tuning knobs we have at our disposal would have encouraged us to tackle our data issues before our neural network problems and helped us to debug more efficiently, instead of running around in circles.
