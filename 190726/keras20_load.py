#1. 데이터 구성
import numpy as np 

x = np.array([range(1000), range(3110, 4110), range(1000)]) 
y = np.array([range(5010, 6010)])

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state = 66, test_size = 0.5)

#2. 모델구성
from keras.models import load_model

model = load_model("savetest01.h5")

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir = './graph', histogram_freq = 0, write_graph = True, write_images = True)

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'auto')

model.fit(x_train, y_train, epochs = 100, batch_size = 8, validation_data = (x_val, y_val), callbacks = [early_stopping, tb_hist])

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 8)
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

print("loss : ", loss)