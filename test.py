import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Load your image
image_path = "C:/Users/szebi/Pictures/Screenshots/hatos.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize your image to match the input size of your model (28x28 pixels)
resized_image = cv2.resize(image, (28, 28))

# Normalize pixel values to range [0, 1]
normalized_image = resized_image / 255.0

# Reshape image to have correct dimensions for prediction
input_image = normalized_image.reshape(1, 28, 28, 1)

# Load your model
model_file_path = "E:\\Sodoku\\your_model.h5"
loaded_model = load_model(model_file_path)

# Predict the digit
prediction = loaded_model.predict(input_image)

# Print the predicted digit
predicted_digit = np.argmax(prediction)
print("Predicted Digit:", predicted_digit)

# Display the image
plt.imshow(resized_image, cmap="gray")
plt.title("Predicted Digit: {}".format(predicted_digit))
plt.axis("off")
plt.show()
