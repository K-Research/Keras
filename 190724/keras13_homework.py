from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1) # (4, 3, 1)  
print("x.shape : ", x.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(1024, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

# 3. 훈련
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs = 1000)

x_input = array([70, 80, 90])
x_input = x_input.reshape(1, 3, 1)

yhat = model.predict(x_input)
print(yhat)