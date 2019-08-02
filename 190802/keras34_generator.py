from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, __ = train_test_split(X_train, random_state = 66, test_size = 0.995)
Y_train, __ = train_test_split(Y_train, random_state = 66, test_size = 0.995)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), strides = (1, 1), padding = 'same', input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(64, (2, 2), padding = 'same',  activation= 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.02, height_shift_range = 0.02, horizontal_flip = True)
model.fit_generator(data_generator.flow(X_train, Y_train, batch_size = 300), steps_per_epoch = 200, epochs = 200, validation_data = (X_test, Y_test), verbose = 1)

print(X_train.shape)
print(Y_train.shape)

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)

# 모델의 실행
# history =model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 30, batch_size = 200, verbose = 1, verbose = 30, callbacks = [early_stopping_callback], checkpointer)
# history = model.fit(X_train, Y_train, epochs = 12, validation_data = (X_test, Y_test), batch_size = 1, verbose = 1, callbacks = [early_stopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 테스트셋의 오차
# y_vloss = history.history['val_loss']

# 학습셋의 오차
# y_loss = history.history['loss']

# 그래프로 표현
# x_len = numpy.arange(len(y_loss))
# plt.plot(x_len, y_vloss, marker = '.', c = "red", label = 'Testset_loss')
# plt.plot(x_len, y_vloss, marker = '.', c = "blue", label = 'Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
# plt.legent(loc = 'upper right')
# plt.axis([0, 20, 0, 0.35])
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()