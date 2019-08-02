# -*- coding : utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
# seed = 0
# numpy.random.seed(seed)
# tf.set_random_seed(seed)

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 # X_train.shape[0] : 60000
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# print(Y_train.shape)
# print(Y_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape)
# print(X_test.shape)

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv2D, Flatten
import numpy as np

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (28, 28, 1), name = 'input')
    x = Conv2D(32, kernel_size = (3, 3), activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(0.2)(x)
    x1 = Conv2D(64, kernel_size = (3, 3), activation = 'relu', name = 'hidden2')(x)
    x1 = Dropout(0.2)(x1)
    x2 = Conv2D(128, kernel_size = (3, 3), activation = 'relu', name = 'hidden3')(x1)
    x2 = Dropout(0.2)(x2)
    x2 = Flatten()(x2)
    prediction = Dense(10, activation = 'softmax', name = 'output')(x2)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers, "keep_prob" : dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함.
# from keras wrappers.scikit_learn import KerasRegressor # 사이킷런과 호환하도록 함.
model = KerasClassifier(build_fn = build_network, verbose = 1) # verbose = 0

model.fit(X_train, Y_train, batch_size = 30, epochs = 10)

print('\n Test Accuracy : %.4f' % (model.evaluate(X_test, Y_test)[1]))