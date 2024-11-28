from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

model = load_model("digit_recognizer.h5")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

test_image = x_test[2].reshape(1, 28, 28, 1)  # Replace with actual MNIST test data
prediction = model.predict(test_image)
print(f"Predicted: {np.argmax(prediction)}, Actual: {y_test[2]}")

incorrect = 0
for i in range(1000):
    test_image = x_test[i].reshape(1, 28, 28, 1)  # Replace with actual MNIST test data
    prediction = model.predict(test_image)
    if np.argmax(prediction) != y_test[i]:
        print(f"Predicted: {np.argmax(prediction)}, Actual: {y_test[2]}")
        incorrect = incorrect+1
print("percent error: ")
print(incorrect / 1000 * 100)