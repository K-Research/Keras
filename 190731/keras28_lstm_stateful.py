import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

# 1. 데이터 수집
a = np.array(range(1, 101))
batch_size = 1
size = 5

def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
# print("=================")
# print(dataset)
# print(dataset.shape)

x_train = dataset[ : , 0 : 4]
y_train = dataset[ : , 4]

x_train = np.reshape(x_train, (len(x_train), size - 1, batch_size))

x_test = x_train + 100
y_test = y_train + 100

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(x_test[0])

# 2. 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful = True))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

num_epochs = 100

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir = './graph', histogram_freq = 0, write_graph = True, write_images = True)

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'auto')

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2, shuffle = False, validation_data = (x_test, y_test), callbacks = [early_stopping, tb_hist])
    model.reset_states()

mse, _ = model.evaluate(x_train, y_train, batch_size = 1)
print("mse : ", mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size = 1)

print(y_predict[0 : 5])

# RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

import matplotlib.pyplot as plt

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc = 'upper left')
plt.axis([0, 20, 0, 0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()