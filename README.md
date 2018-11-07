# Robot Learning: Person Detection
## Project Goal
The goal of our project was to use the neato’s camera to interact with a human and play the game of tag. The game starts with a state machine that qualifies the neato as either “it” or “not it.” The mode dictates whether or not the neato is actively chasing the person in its view or driving away. Switching between the modes is toggled with the bump sensor.
To process the data the camera would is first used to gather a training set of images and build a model for mapping a new image to a person’s position. The model is downloaded and used live on the neato for locating the person and adjusting its linear and angular velocities appropriately.

[KATYA DO]How did you solve the problem?  You should touch on the data you used, the algorithms you applied, etc.  If you implemented your own version of an algorithm as a step in your learning process, make sure to explain what you did for that part of the project as well.
***** Talk about the neural net architecture. Talk about the data processing algorithm.

## Design Decisions
One design decision we made was using the LIDAR to classify the 2D position of the person its view. The root decision here was to attempt to locate the person in the room’s spatial coordinate instead of the image’s 2D coordinates. This gave us a way to account how the person’s depth changes in the image as they move around the space.
	Our decision to use the LIDAR for labelling our training set was with the goal of automating and parallelizing our image recording and classification. This was made possible because both our data and labelling was done with ROS topics running on neato-hardware. In an ideal world this would make accurate data collection in large batches fast, simple processes.

## Challenges
- We tried to start by training the model to identify a person in the image’s 2D space. We thought this would be a helpful start to the project, but bad mouse-tracking data and a generally poor data set for this type of relationship led to frustrating results that never made any valuable predictions.
- We tried a couple different approaches to data logging and classifying the center of mass. This should have been much more simple but was not porting over from the warmup project as successfully as we would have hoped and led to delays.
- We pivoted to focus 100% on the lidar-based classification and ran into many challenges related to image classification. A critical error was “interpolated_scan” topic being unable to function correctly. This created the subsequent challenge of correlating lidar data and image data that were recorded at different frequencies. 

## Future Improvements
- Experiment with different classification methods and tuning our dataset to be as clear and correlative to our labels as possible.
- If our model was functional we would have dove further into the robot implementation. There are many design decisions and levels of depth we could have taken the on-robot implementation, and having time to experiment with and develop this part of the project would have been another exciting challenge to handle.

## Lessons Learned
We learned to value many key processes in robotic development.
- Verifying and understanding large datasets. Building algorithms for both robust data collection, processing, and verification at the very beginning of the project would have saved us a ton of time at the end when we needed time to tune our neural net or rapidly take sets of data.
