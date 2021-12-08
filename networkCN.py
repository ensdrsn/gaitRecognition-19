# -*- coding: utf-8 -*-
"""combined.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lbMuU0gSyzNyKuqeHfIlTGvFTG2xTKGj
"""

from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D, Flatten, Reshape, Conv2D
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
import time
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from keras.regularizers import l2
from keras.utils import plot_model
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive/')

X = np.load('/content/drive/My Drive/GaitRecognition-Data/X_train_f.npy', allow_pickle=True)
y = np.load('/content/drive/My Drive/GaitRecognition-Data/y_train_f.npy', allow_pickle=True)
X_test = np.load('/content/drive/My Drive/GaitRecognition-Data/X_test_f.npy', allow_pickle=True)
y_test = np.load('/content/drive/My Drive/GaitRecognition-Data/y_test_f.npy', allow_pickle=True)

X = X/255.0
X_test = X_test/255.0



no_of_img = 50
img_width = 32
img_height = 32
num_classes = 124
channel = 11
batch_sizes = [64]
kernel_sizes = [(3, 3)]
layer_sizes = [32]
conv_layers = [2]
lrs =[0.0001]

for conv_layer in conv_layers:
    for layer_size in layer_sizes:
      for batch_size in batch_sizes:
        for lra in lrs:
          for kernel_size in kernel_sizes: 

            clock = time.strftime('%d%m%Y-%H%M%S')
            NAME = "Gait-LSTM-All-{}-conv-{}-nodes-{}-batch-{}-lr-{}-kernel-{}".format(conv_layer, layer_size, batch_size, lra, kernel_size, clock)
            
            print(NAME)
            model = Sequential()

            model.add(ConvLSTM2D(layer_size, padding='same', kernel_size=kernel_size, activation='relu',
                                 recurrent_activation='tanh', input_shape=(no_of_img, img_width, img_height, channel),
                                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                                 data_format='channels_last', return_sequences=True))
            model.add(BatchNormalization())
            model.add(TimeDistributed(MaxPooling2D((2, 2))))
            model.add(Dropout(0.5))

            model.add(ConvLSTM2D(layer_size, padding='same',kernel_size=kernel_size, activation='relu',
                                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                                 recurrent_activation='tanh', return_sequences=False))
            model.add(BatchNormalization())
            model.add((MaxPooling2D(pool_size=(2, 2))))
            model.add(Dropout(0.5))

            model.add(Flatten())

            model.add(Dense(2048, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            
            model.add(Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(num_classes, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Activation('softmax'))

            opt = optimizers.adam(lr=lra, decay=1e-5)

            model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            model.summary()
            plot_model(model, to_file='/content/drive/My Drive/GaitRecognition-Data/{}.png'.format(NAME), show_shapes=True)

            history = model.fit(X, y, epochs=100, batch_size=batch_size, shuffle=True)
            
            score = model.evaluate(X_test, y_test, batch_size=batch_size)
            
            plt.plot(history.history['acc'])
            plt.title('Model training accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig('/content/drive/My Drive/GaitRecognition-Data/Tr-acc-{}-testacc-{}.png'.format(score[1], NAME))
            plt.show()   
            
            plt.plot(history.history['loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig('/content/drive/My Drive/GaitRecognition-Data/Tr-loss-{}-testloss-{}.png'.format(score[0], NAME))
            plt.show()

            print('Test loss:', score[0])
            print('Test accuracy:', score[1])