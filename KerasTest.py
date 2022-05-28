import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

# num_classses : This variable defines
# the number of classes or the emotions that
# we will be dealing with in training our model.
num_classes = 5

# img_rows,img_cols=48,48 : These variables define
# the size of the image array that
# we will be feeding to our neural network.
img_rows, img_cols = 48, 48

# batch_size=32: This variable defines
# the batch size.The batch size is a number of samples processed
# before the model is updated.
# The number of epochs is the number of complete passes
# through the training dataset. The size of a batch must be
# more than or equal to one and less than or equal to the number of samples
# in the training dataset.
batch_size = 32


