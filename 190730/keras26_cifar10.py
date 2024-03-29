from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CIFAR_10은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수 정의
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape : ', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 범주형으로 전환
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 실수형으로 지정하고 정규화
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.flatten()
X_test = X_test.flatten()
X_train = X_train.reshape(X_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.flatten()
X_test = X_test.flatten()

X_train = X_train.reshape(-1, 3)
X_test = X_test.reshape(-1, 3)

X_train = X_train.reshape(-1, 32, 3)
X_test = X_test.reshape(-1, 32, 3)

X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# 신경망 정의
model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()

# 학습
model.compile(loss = 'categorical_crossentropy', optimizer = OPTIM, metrics = ['accuracy'])

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir = './graph', histogram_freq = 0, write_graph = True, write_images = True)

from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)

history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, validation_split = VALIDATION_SPLIT, verbose = VERBOSE, callbacks = [early_stopping_callback, tb_hist])

print('Testing...')
score = model.evaluate(X_test, Y_test, batch_size = BATCH_SIZE, verbose = VERBOSE)
print("\nTest scroe : ", score[0])
print('Test accuracy : ', score[1])

# 모델 저장
# model.json = model.to_json()
# open('cifar10_architecture.json', 'w').write(model_json)
# model.save_weights('cifar10_weights.h5', overwrite = True)

# 히스토리에 있는 모든 데이터 나열
print(history.history.keys())

# 단순 정확도에 대한 히스토리 요악
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

# 손실에 대한 히스토리 요약
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()