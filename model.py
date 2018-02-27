import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import matplotlib.image as mpimg
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.callbacks import ModelCheckpoint

# Read the CSV file
def readFileNames():
    samples =[]
    with open ('./data_t2/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        return samples   

    
# Generator for feeding data to the model

def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #Process all images of the current line
                names =[]
                names.append( './data_t2/IMG/'+batch_sample[0].split('\\')[-1])  #center
                names.append( './data_t2/IMG/'+batch_sample[1].split('\\')[-1])  #left
                names.append( './data_t2/IMG/'+batch_sample[2].split('\\')[-1])  #right
                
                correction_factor =[0,0.25,-0.25]
                
                for i in range(len(names)):
                    name = names[i]
                    correction = correction_factor[i]
                    image = mpimg.imread(name)
                    angle = float(batch_sample[3])+correction
                    # Push current image & angle
                    images.append(image)
                    angles.append(angle)
                    image_flip = cv2.flip( image, 1 )
                    angle = angle*-1.0
                    
                    # Push Flipped image & angle
                    images.append(image_flip)
                    angles.append(angle)

            # Keras accepts only numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def makeModel(ch=3, row=160, col=320, top_crop = 0, botton_crop = 0):
    model = Sequential()
    # Add the Normalization layer
    model.add(Lambda(lambda x: x/255.0 -0.5,
                     input_shape=( row, col,ch),
                    output_shape=( row, col,ch)))
    
    # Crop the image to remove distractions
    model.add(Cropping2D(cropping=((top_crop,botton_crop), (0,0))))
    
    # 5 Convolution layers
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    
    # Flatten
    model.add(Flatten())
    model.add(Dropout(0.5)) # added dropout to nVedia model
    
    # 3 fully connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    
    # Output
    model.add(Dense(1)) 
    return model

# Read File Names
samples = readFileNames()
# Split traning and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# Make generator for traning and validation set
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
# Get an instance of nVidia model
model = makeModel(3,160,320, top_crop = 50, botton_crop= 20)
# compile with adam optimizer and use Mean Square Error as loss function. 
model.compile(loss='mse', optimizer='adam')
# Save the model with least validation loss
checkpoint_callback = ModelCheckpoint('best_nVidia.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# Train the model
model.fit_generator(train_generator, 
                    samples_per_epoch= len(train_samples)*6, 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=4,callbacks=[checkpoint_callback])