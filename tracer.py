import cv2
import numpy as np
import tensorflow as tf
print("hi")
from tensorflow import keras

# Load your pretrained sign language detection model.
print("hi")
model = keras.models.load_model("C:\\Users\\saisu\\OneDrive\\Desktop\\proj\\sign_lang\\sign_language_detection_model.h5")
print("hi")


# Initialize the webcam.
cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust if needed.

while True:
    # Capture a frame from the webcam.
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame to match the input size of your model.
    frame = cv2.resize(frame, (64, 64))  # Adjust the size as per your model's input size.
    frame = frame / 255.0  # Normalize pixel values to [0, 1]

    # Perform inference using your model.
    prediction = model.predict(np.expand_dims(frame, axis=0))

    # Get the predicted class label (gesture).
    predicted_class = np.argmax(prediction)

    # Display the frame and the predicted class label.
    print("hi",predicted_class)
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
