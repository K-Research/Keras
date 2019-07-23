import numpy as np

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1, activation = 'relu')) # model.add : layer 추가, Dense : Node의 수
model.add(Dense(3)) # Node : 3
model.add(Dense(4)) # Node : 4
model.add(Dense(1)) # Node : 1

# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 100, batch_size = 1) # model.fit : model 실행, epochs : 반복 횟수, batch_size = 1 : 한 개씩 잘라서 훈련

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print("acc : ", acc)