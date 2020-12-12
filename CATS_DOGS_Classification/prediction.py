import keras
import cv2
import numpy as np

def loadImages(path1,path2):

    #loading images to test
    BGR = cv2.imread(path1)
    BGR2 = cv2.imread(path2)

    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    RGB2 = cv2.cvtColor(BGR2, cv2.COLOR_BGR2RGB)
    image = cv2.resize(RGB,(150,150))
    image2 = cv2.resize(RGB2,(150,150))
    #print(image/255)
    #print(image2/255)
    test = [image,image2]
    nptest= np.array(test)
    print(np.shape(nptest))

    return nptest

def main():
    test = loadImages("test1.jpg","test2.jpg")
    # loading the model
    model = keras.models.load_model('Model_cats_dogs_small_dataset.h5')
    print(model.predict(test))

if __name__ == '__main__':
    main()