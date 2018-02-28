# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/0center.png "center image"
[image2]: ./examples/0center_flipped.png "center flipped image"
[image3]: ./examples/0left.png "left image"
[image4]: ./examples/0left_flipped.png "left flipped image"
[image5]: ./examples/0right.png "right image"
[image6]: ./examples/0right_flipped.png "right flipped image"
[image8]: ./examples/2000center.png "center image"
[image9]: ./examples/2000center_flipped.png "center flipped image"
[image10]: ./examples/2000left.png "left image"
[image11]: ./examples/2000left_flipped.png "left flipped image"
[image12]: ./examples/2000right.png "right image"
[image13]: ./examples/2000right_flipped.png "right flipped image"
[image15]: ./examples/4000center.png "center image"
[image16]: ./examples/4000center_flipped.png "center flipped image"
[image17]: ./examples/4000left.png "left image"
[image18]: ./examples/4000left_flipped.png "left flipped image"
[image19]: ./examples/4000right.png "right image"
[image20]: ./examples/4000right_flipped.png "right flipped image"
[image22]: ./examples/6000center.png "center image"
[image23]: ./examples/6000center_flipped.png "center flipped image"
[image24]: ./examples/6000left.png "left image"
[image25]: ./examples/6000left_flipped.png "left flipped image"
[image26]: ./examples/6000right.png "right image"
[image27]: ./examples/6000right_flipped.png "right flipped image"
[image29]: ./examples/8000center.png "center image"
[image30]: ./examples/8000center_flipped.png "center flipped image"
[image31]: ./examples/8000left.png "left image"
[image32]: ./examples/8000left_flipped.png "left flipped image"
[image33]: ./examples/8000right.png "right image"
[image34]: ./examples/8000right_flipped.png "right flipped image"
[image36]: ./examples/10000center.png "center image"
[image37]: ./examples/10000center_flipped.png "center flipped image"
[image38]: ./examples/10000left.png "left image"
[image39]: ./examples/10000left_flipped.png "left flipped image"
[image40]: ./examples/10000right.png "right image"
[image41]: ./examples/10000right_flipped.png "right flipped image"
[image43]: ./examples/12000center.png "center image"
[image44]: ./examples/12000center_flipped.png "center flipped image"
[image45]: ./examples/12000left.png "left image"
[image46]: ./examples/12000left_flipped.png "left flipped image"
[image47]: ./examples/12000right.png "right image"
[image48]: ./examples/12000right_flipped.png "right flipped image"
[image49]: ./examples/Capture.PNG  "track1 forward" 
[image50]: ./examples/Capture2.PNG "track1 backward"
[image51]: ./examples/Capture3.PNG "track2 forward"
[track1_video.mp4]:  ./track1_video.mp4
[track2_video.mp4]:  ./track2_video.mp4

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

UPDATE: I tested this theroy using aws workspace (with GPU) and the model was able to drive both tracks with max speed set to 20.


#### 2. Final Model Architecture

The final model architecture (model.py lines 64-92) consisted of a convolution neural network with the following layers and layer sizes.

```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 8448)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I changed direction and recorded two laps on track one also using center lane driving. 

![alt text][image49] ![alt text][image50]

Then I recorded two laps on track two.


![alt text][image51]

I did not record recovery data seperatly since I had decided to use left and right cameras and their flipped images for that purpose. If flipping images was not sufficiant for making recovery data I would go back and record some but thankfully it was not needed.
A correction factor of 0.25 for left camera and -0.25 for right camera was used. This produced enough recovery data for the model to keep the vehicle from going off the road.

Center image 

![alt text][image8]![alt text][image1]

Left camera

![alt text][image10]![alt text][image3]

Right camera

![alt text][image12] ![alt text][image5]



After the collection process, I had 13824 data points and each point had 3 images (center,left & right). I preprocessed the data by flipping the images and multiplying the associated stearing angle by -1. This not only increased the amount of data we have for traning but also balanced the images so that there was no bias for turning in either direction.

Center image and its flipped version

![alt text][image43]![alt text][image44]

Left image and its flipped version

![alt text][image45]![alt text][image46]

Right image and its flipped version

![alt text][image47]![alt text][image48]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for 10 epochs and used callbacks to save the model with the lowest validation loss.

The model was successfully able to drive the vehicle on both the first and second track. The tire did not leave the drivable portion of the track surface at any point on both the tracks.

**video for Track 1 :** [track1_video.mp4]

**video for Track 2 :** [track2_video.mp4]
