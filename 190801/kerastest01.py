import pandas
from numpy import array
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

dataset_market = pandas.read_csv('kospi.csv', usecols = [1])
dataset_high = pandas.read_csv('kospi.csv', usecols = [2])
dataset_low = pandas.read_csv('kospi.csv', usecols = [3])
dataset_closing = pandas.read_csv('kospi.csv', usecols = [4])
dataset_volume = pandas.read_csv('kospi.csv', usecols = [5])
dataset_exchange = pandas.read_csv('kospi.csv', usecols = [6])

x_train_market = dataset_market
x_train_high = dataset_high
x_train_low = dataset_low
x_train_volume = dataset_volume
x_train_exchange = dataset_exchange

x_train_market = x_train_market.values.reshape(x_train_market.shape[0], x_train_market.shape[1])
x_train_high = x_train_high.values.reshape(x_train_high.shape[0], x_train_high.shape[1])
x_train_low = x_train_low.values.reshape(x_train_low.shape[0], x_train_low.shape[1])
x_train_volume = x_train_volume.values.reshape(x_train_volume.shape[0], x_train_volume.shape[1])
x_train_exchange = x_train_exchange.values.reshape(x_train_exchange.shape[0], x_train_exchange.shape[1])

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train_market)
x_train_market = scaler.transform(x_train_market)
x_train_high = scaler.transform(x_train_high)
x_train_low = scaler.transform(x_train_low)
x_train_volume = scaler.transform(x_train_volume)
x_train_exchange = scaler.transform(x_train_exchange)

x_train = numpy.concatenate((x_train_market, x_train_high, x_train_low, x_train_volume, x_train_exchange), axis = 1)

x_train = x_train.reshape(599, 5, 1)

y_train = dataset_closing

x_input = array([2026.10, 2032.23, 2009.33, 284936, 1187.80])
x_input = x_input.reshape(1, 5, 1)

model = Sequential()

model.add(LSTM(input_dim = 1, output_dim = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(output_dim = 1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('Compilation time : ', time.time() - start)

# model = Sequential()
# model.add(LSTM(input_dim = 1, output_dim = 50, return_sequences = True))
# model.add(LSTM(100, return_sequences = False))
# model.add(Dense(output_dim = 1))
# model.add(Activation('linear'))

model.compile(loss = 'mse', optimizer = 'rmsprop')

model.fit(x_train, y_train, batch_size = 1, nb_epoch = 100, validation_split = 0.05)

yhat = model.predict(x_input)
print(yhat)