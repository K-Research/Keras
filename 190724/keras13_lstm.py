from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = array([4, 5, 6, 7])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

X = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape : ", x.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(5))
model.add(Dense(1))