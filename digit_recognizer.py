import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('mnist_digit_model.h5')

def is_empty(cell_image, threshold=0.95):
    """Egy cella ürességének ellenőrzése a pixelintenzitás alapján."""
    if cell_image.ndim == 3:
        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    cell_image = cv2.resize(cell_image, (28, 28))
    non_empty_pixels = np.count_nonzero(cell_image < 255 * threshold)
    total_pixels = cell_image.shape[0] * cell_image.shape[1]
    if (non_empty_pixels / total_pixels) < threshold:
        return True
    return False

def preprocess_for_model(cell_image):
    if cell_image is None or np.sum(cell_image) == 0:
        return None  # Check for empty or completely white image

    # Convert to grayscale if it is a colored image
    if len(cell_image.shape) == 3:
        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    cell_image = cv2.resize(cell_image, (28, 28))

    # Normalize the pixel values (0-255 to 0-1)
    cell_image = cell_image / 255.0

    # Reshape to fit the model input
    cell_image = cell_image.reshape(1, 28, 28, 1)  # Batch size, width, height, channels

    return cell_image


def recognize_digit(cell_image):
    # Preprocess the image
    preprocessed_image = preprocess_for_model(cell_image)
    if preprocessed_image is None:
        return 0  # Assume empty if preprocessing fails or is empty

    # Predict the digit using the model
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)

    # Optional: check confidence
    confidence = np.max(prediction)
    if confidence < 0.8:  # Confidence threshold to avoid mispredictions
        return 0

    return predicted_digit

