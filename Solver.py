from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28
num_classes = 10


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train /= 255
x_test /= 255


image_index = 35
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()
print(x_train.shape)
print(x_test.shape)
print(y_train[:image_index + 1])