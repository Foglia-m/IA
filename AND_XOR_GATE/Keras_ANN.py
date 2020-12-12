
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def defineModel(input_shape, num_classes):

    model = Sequential()


    model.add(Dense(8, input_dim = input_shape,activation='tanh'))


    model.add(Dense(num_classes, activation='sigmoid'))


    model.compile(optimizer = 'SGD', loss='binary_crossentropy', metrics=['accuracy'])


    return model

def trainAndEvaluateClassic(trainX, trainY):


    # Calling the defineModel function
    model = defineModel(2, 1)

    #fitting the data
    model.fit(trainX, trainY, batch_size= 1, epochs=800,verbose=1)

    #evaluation
    print("input = \n" + str(trainX) +"\n" + " prediction = \n" + str(model.predict(trainX)) +"\n" + " actual label = \n" + str(trainY))
    return


def main():



    #Input data
    points = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    #Labels
    labels = np.array([[0], [1], [1], [0]])

    trainAndEvaluateClassic(points, labels)



if __name__ == '__main__':
    main()