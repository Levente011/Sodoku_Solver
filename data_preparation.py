from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Adatok átalakítása és normalizálása
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

    # Címkék one-hot kódolása
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)
