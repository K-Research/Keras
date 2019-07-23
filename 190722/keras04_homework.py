import numpy as np

# 1. 데이터
x_train = np.array([])
for i in range(1, 101):
    x_train = np.append(x_train, i)
y_train = np.array([])
for i in range(501, 601):
    y_train = np.append(y_train, i)
x_test = np.array([])
for i in range(1001, 1101):
    x_test = np.append(x_test, i)
y_test = np.array([])
for i in range(1101, 1201):
    y_test = np.append(y_test, i)

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(100, input_dim = 1, activation = 'relu')) # model.add : layer 추가, Dense : Node의 수
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

# model.summary()

# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 1000, batch_size = 3)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 3)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)