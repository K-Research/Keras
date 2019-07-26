#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
model = Sequential()

from keras import regularizers

model.add(Dense(1000, input_shape = (3, ), activation = 'relu', kernel_regularizer = regularizers.l1(0.1)))
# model.add(BatchNormalization)
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

model.save('savetest01.h5')
print("저장이 되었습니다.")