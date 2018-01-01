import glob
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
import keras
import numpy as np
from sklearn.model_selection import train_test_split

cars = glob.glob('data/vehicles/*/*')
notcars = glob.glob('data/non-vehicles/*/*')

X_raw = []
y_raw = []

for car in cars:
    image = cv2.imread(car)
    if image.shape == (64, 64, 3):
        image = image.astype(np.float32) / 255 - 0.5
        X_raw.append(image)
        y_raw.append(1)

for car in notcars:
    image = cv2.imread(car)
    if image.shape == (64, 64, 3):
        image = image.astype(np.float32) / 255 - 0.5
        X_raw.append(image)
        y_raw.append(0)

y_raw = keras.utils.np_utils.to_categorical(y_raw)
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=rand_state)

model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

model.fit(np.array(X_train), y_train,
          verbose=1,
          batch_size=100,
          nb_epoch=30,
          validation_split=0.2,
          shuffle=True)


scores = model.evaluate(np.array(X_test), y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save("cnn.h5")
