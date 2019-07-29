from keras.models import Sequential

filter_size = 32
kernel_size = (3, 3)

from keras.layers import Conv2D, MaxPooling2D
model = Sequential()
model.add
# model.add(Conv2D(filter_size, kernel_size, input_shape = (28, 28, 1)))
model.add(Conv2D(7, (2, 2), padding = 'same', input_shape = (5, 5, 1)))
model.add(Conv2D(16, (2, 2)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(8, (2, 2)))

model.summary()