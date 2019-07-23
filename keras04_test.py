import numpy as np

# 1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(1000, input_dim = 1, activation = 'relu')) # model.add : layer 추가, Dense : Node의 수
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 3)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)