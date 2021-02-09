import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils


# VARIABLES
NB_EPOCH = 20 # number of times it loops through the whole data set
BATCH_SIZE = 128 # number of images the ai is given per epoch (shuffled randomly)
VERBOSE = 1
NB_CLASSES = 10 # number of outputs
OPTIMIZER = Adam() 
N_HIDDEN = 128 # number of nuerons per hidden layer
VALIDATION_SPLIT = .2 # percentage of training data set split for validation
HIDDEN_LAYERS = 2 # number of layers between input layer and output layer
DROPOUT = .3 # percentage of neurons randomly ignored (form of regulization)

(X_train, y_train), (X_test, y_test) = mnist.load_data() # load MNIST data
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED) # flatten input array (for training and testing) and define input type
X_test = X_test.reshape(10000, RESHAPED) 
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # normalize (grayscale values to floats between 0 and 1)
X_test /= 255
print(X_train.shape[0], 'train samples') 
print(X_test.shape[0], 'test samples')

# labels to categorical arrays (same thing as one hot encoded in the NN I wrote from scratch)
Y_train = np_utils.to_categorical(y_train, NB_CLASSES) 
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# initialize squential model
model = Sequential()

# add hidden layers with reLU activation
for i in range(HIDDEN_LAYERS):
	model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
	model.add(Activation('relu'))
	model.add(Dropout(DROPOUT))

# add output layer with softmax activation function
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))

model.summary()

# compile model with all the variables/preproccessed data and define metrics
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, \
	epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# evaluate model (give it training images)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

#
print("Test score:", score[0])
print("Test accuracy:", score[1])