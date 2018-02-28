# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video_t1.mp4 contaning a video made from center camera images captured while driving autonomously in track one.
* video_t2.mp4 contaning a video made from center camera images captured while driving autonomously in track two. 


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with either 3x3 or 5x5 filter sizes and depths between 24 and 64 (model.py lines 64-92) 
The model includes RELU layers to introduce nonlinearity (code line 75-79), a cropping layer (code line 72 ) and the data is normalized in the model using a Keras lambda layer (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting. Dropout is introduced just after the first fully connected layer so that it is not too dependent on a particular feature from the final convolution layer to make predictions (model.py lines 83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97 & 108). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

While collecting traning data I tried driving as close to the center of the road as possible. Recovery data was obtained by using the left and right camera images and their flipped versions.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a simple neural network with two fully connected layers to test if everything was working properly i.e if i could train a model and use sumilator to test the model in autonomous mode. After successfuly testing the setup I decided to go with nVidia's model which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers.
I also added a crop layer after normalization.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I got a low test loss as well as a low validation loss (although not as low). Although I didn't think my modle was overfitting, I inserted Dropout after flattening which did reduce the validation loss further.    

The final step was to run the simulator to see how well the car was driving around track one. The vehicle didnt stay on track and went off the road very quickly. I tried changing model architecture but it didn't help.I reduced the speed at which the car drives in drive.py file and after that the vehicle drove autonomously several times around the track. I think the issue was that my PC was not able to process the data fast enough to keep the vehicle on track. 

I think if I run the model on a PC with better hardware it should be able to drive autonomously without lowering the max speed.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-92) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image 							|
| Normalization         		|  							|
| Crop     		|  							|
| Convolution 5x5     	| 2x2 stride, filters 24, outputs 78x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, filters 36, outputs 37x156x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, filters 48, outputs 17x76x48 	|
| RELU					|												|
| Convolution 3x3     	| 2x2 stride, filters 64 ,outputs 6x36x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, filters 64 ,outputs 4x34x64 	|
| RELU					|												|
|Flatten|           									|
| Dropout		|       Drop prob 0.5 								|
| Fully connected 				| input 8704 , output 100       									|
| Fully connected 				| input 100 , output 50       									|
| Fully connected 				| input 50 , output 10       									|
| output|Vehicle control  |


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I changed direction and recorded two laps on track one also using center lane driving. 

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded two laps on track two.


![alt text][image6]
![alt text][image7]

I did not record recovery data seperatly since I had decided to use left and right cameras and their flipped images for that purpose. If flipping images was not sufficiant for making recovery data I would go back and record some but thankfully it was not needed.

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for 10 epochs and used callbacks to save the model with the lowest validation loss.

The model was successfully able to drive the vehicle on both the first and second track. The tire did not leave the drivable portion of the track surface at any point on both the tracks.
