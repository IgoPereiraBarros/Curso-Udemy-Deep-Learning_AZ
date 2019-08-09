# -*- coding: utf-8 -*-

# Part 1 - Building the CNNs

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#  Iniatilising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Tuning the ANN - Adding a secund convulutional layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
               steps_per_epoch=8000,
               epochs=25,
               validation_data=test_set,
               validation_steps=2000)


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

new_img = load_img(path='dataset/single_prediction/predict/g5.jpeg', target_size=(64, 64))
new_img = img_to_array(new_img)
new_img = np.expand_dims(new_img, axis=0)
result = classifier.predict(new_img)

if training_set.class_indices['cats'] == result:
    pred = 'Cat'
else:
    pred = 'Dog'

