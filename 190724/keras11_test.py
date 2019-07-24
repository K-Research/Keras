#1. 데이터 구성
import numpy as np 

x = np.array([range(100), range(311, 411), range(401, 501)]) 
y = np.array([range(501, 601)])

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state = 66, test_size = 0.5)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential()

model.add(Dense(1111, input_shape = (3, ), activation = 'relu'))
model.add(Dense(1111))
model.add(Dense(1111))
model.add(Dense(1))

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data = (x_val, y_val))

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)
print("acc : ", acc)
y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)