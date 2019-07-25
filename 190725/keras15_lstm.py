import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1, 11))

size = 5

def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("=====================")

x_train = dataset[:, 0 : 4]
y_train = dataset[:, 4, ]

print(x_train.shape)
print(y_train.shape)