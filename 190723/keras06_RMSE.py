import numpy as np

# 1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x3 = np.array([101, 102, 103, 104, 105, 106])
x4 = np.array(range(30, 50))

from keras.models import Sequential
from keras.layers import Dense

# 2. 모델구성
model = Sequential()
# model.add(Dense(5, input_dim = 1, activation = 'relu')) # input_dim = 열의 갯수
model.add(Dense(64, input_shape = (1, ), activation = 'relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs = 100, batch_size = 3)
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 3)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))