from keras.utils import np_utils
from keras.datasets import fashion_mnist
import cv2
from keras import models
from matplotlib import pyplot
#load the image:
def load_image(file):
    im = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im,(28,28)) # resizing
    im = im / 255 # normalizing
    im = im.reshape(1,28,28,1)
    return im


model = models.load_model('Fashion_MNIST_model.h5')
print(model.predict(load_image("sample_image.png")))
