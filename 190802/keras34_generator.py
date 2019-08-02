from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.02, height_shift_range = 0.02, horizontal_flip = True)
# model.fit_generator(data_generator.flow(X_train, Y_train, batch_size = 32), steps_per_epoch = len(X_train) // 32, epochs = 200, validation_data = X_test, Y_test, verbose = 1, callbacks = callbacks)
model.fit_generator(data_generator.flow(X_train, Y_train, batch_size = 32), steps_per_epoch = len(X_train) // 32, epochs = 200, validation_data = X_test, Y_test, verbose = 1)