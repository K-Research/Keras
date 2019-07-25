import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler

a = np.array(range(1, 11))

size = 5

def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset = split_5(a, size)

x_train = dataset[:, 0 : 4]
y_train = dataset[:, 4]

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)

# x_train = np.reshape(x_train, (6, 4, 1))
x_train = np.reshape(x_train_scaled, (len(a) - size + 1, 4, 1))

x_test = np.array([[[11], [12], [13], [14]], [[12], [13], [14], [15]], [[13], [14], [15], [16]], [[14], [15], [16], [17]]])
y_test = np.array([15, 16, 17, 18])

x_test_scaled = scaler.transform(x_test[:, :, 0])
x_test_scaled = np.reshape(x_test_scaled, (4, 4, 1))

# 2. 모델 구성
model = Sequential()

model.add(LSTM(32, input_shape = (4, 1), return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(10))

model.add(Dense(5, activation = 'relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


model.summary()

# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.fit(x_train, y_train, epochs = 10000, batch_size = 1, verbose = 1, callbacks = [early_stopping])

loss, acc = model.evaluate(x_test_scaled, y_test)

y_predict = model.predict(x_test_scaled)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)