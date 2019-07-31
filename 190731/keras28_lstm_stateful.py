import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

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
print("=================")
print(dataset)
print(dataset.shape)

x_train = dataset[ : , 0 : 4]
y_train = dataset[ : , 4]

x_train = np.reshape(x_train, (len(x_train), size - 1, batch_size))

x_test = x_train * 2
y_test = y_train * 2

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])

# 2. 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful = True))
model.add(Dense(1))

model.summary()

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

num_epochs = 50

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2, shuffle = False, validation_data = (x_test, y_test))
    model.reset_states()

mse, _ = model.evaluate(x_train, y_train, batch_size = 1)
print("mse : ", mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size = 1)

print(y_predict[0 : 10])