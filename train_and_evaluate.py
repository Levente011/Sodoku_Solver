import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Adathalmaz betöltése és előkészítése
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizálás és átalakítás a modell számára megfelelő formátumra
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Címkék one-hot encodingja
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Modell létrehozása
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Modell fordítása
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modell tanítása
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

# Modell értékelése
score = model.evaluate(x_test, y_test, verbose=0)
print('Teszt veszteség:', score[0])
print('Teszt pontosság:', score[1])

# Modell mentése
model.save('mnist_digit_model.h5')
