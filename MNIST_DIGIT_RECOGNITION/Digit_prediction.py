import keras
import cv2
import numpy as np
from keras import models
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from matplotlib import pyplot



def baseline_model(num_pixels, num_classes):

    #Application 1 - Step 5 - Initialize the sequential model
    model = Sequential()


    model.add(Dense(8, input_dim=num_pixels, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal',
                    activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


    return model


def main():


    #loading the mnist data as test_images did not work

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # TODO - Application 1 - Step 2 - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

    # TODO - Application 1 - Step 3 - Normalize the input values
    X_train = X_train / 255
    X_test = X_test / 255

    # TODO - Application 1 - Step 4 - Transform the classes labels into a binary matrix
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]


    #creating the new model
    model = baseline_model(num_pixels,num_classes)
    #loading the weights
    model.load_weights("cnn.cktp")
    #doesnotexist = mnist.test_images()
    #testing
    for i in range (5):
        image= X_test[i]
        print(np.argmax(model.predict(image)))

    return

if __name__ == '__main__':
    main()