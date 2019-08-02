import pandas
from numpy import array
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy
from sklearn.model_selection import train_test_split

kospi = pandas.read_csv('kospi.csv')
dataset_closing_price = array(kospi["Closing price"])

size = 4

def split_4(seq, size):
    list = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        list.append(subset)
    return list

dataset = split_4(dataset_closing_price, size)
dataset = array(dataset)

x = dataset[:, 0 : 2]
y = dataset[:, 2 : ]

x = x.reshape(596, 2, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.0001)

model = Sequential()

model.add(LSTM(50, input_shape = (2, 1), return_sequences = True))
model.add(LSTM(100, return_sequences = False))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Activation('linear'))

model.compile(loss = 'mse', optimizer = 'rmsprop')

model.fit(x_train, y_train, batch_size = 1, nb_epoch = 100, validation_split = 0.05)

y_predict = model.predict(x_test)
print(y_predict)