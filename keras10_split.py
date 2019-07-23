# 1. 데이터
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(1, 101))

# x_train = x[ : 60]
# x_val = x[60 : 80]
# x_test = x[80 : 100]
# y_train = x[ : 60]
# y_val = x[60 : 80]
# y_test = x[80 : 100]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state = 66, test_size = 0.25)

print(x_train)
print(x_val)
print(x_test)
print(y_train)
print(y_val)
print(y_test)

# # 2. 모델구성
# from keras.models import Sequential
# from keras.layers import Dense
# model = Sequential()

# model.add(Dense(16, input_dim = 1, activation = 'relu'))
# # model.add(Dense(16, input_shape = (1, ), activation = 'relu'))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))

# # 3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
# model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data = (x_val, y_val))

# # 4. 평가 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size = 3)
# print("acc : ", acc)

# y_predict = model.predict(x_test)
# print(y_predict)

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score

# r2_y_predict = r2_score(y_test, y_predict)
# print("R2 : ", r2_y_predict)