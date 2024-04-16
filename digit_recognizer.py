import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('E:\Sodoku/your_model.h5')

def preprocess_for_model(cell_image):
    """A cella képének előkészítése a modell számára."""
    img = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    img = img.reshape((1, 28, 28, 1))  # Reshape image for the model
    return img



def recognize_digit(cell_image):
    # Preprocess the image
    preprocessed_image = preprocess_for_model(cell_image)
    print("Preprocessed Image Shape:", preprocessed_image.shape)  # Debugging
    print("Preprocessed Image Pixel Range:", preprocessed_image.min(), preprocessed_image.max())  # Debugging

    # Predict the digit using the model
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    print("Model Prediction:", prediction)  # Debugging
    print("Predicted Digit:", predicted_digit)  # Debugging

    return predicted_digit


