# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Endrer shapen fra (28,28) til (28,28,1)
    # fordi Conv2D trenger input (height,width,channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Eksempel: Gjør om '2' til [0,0,1,0,0,0,0,0,0,0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

    # Normaliser dataen fra (0,255) til (-0.5,0.5)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train -= 0.5
x_test -= 0.5

epochs = 1
batch_size = 128

### ----------------------------------------------------------------

model = Sequential()

model.add(Conv2D(32, (2,2), activation='elu', input_shape=(28,28,1), use_bias=False))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (2,2), activation='elu', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (2,2), activation='elu', use_bias=False))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='elu', use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### ----------------------------------------------------------------

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
