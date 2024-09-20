import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your data directory where the labeled images are stored.
data_dir = "C:\\Users\saisu\\OneDrive\\Desktop\\proj\\sign_lang\\images\\other\\imgs"

# Define your image dimensions and batch size.
image_size = (64, 64)
batch_size = 32

# Create an ImageDataGenerator for data augmentation and preprocessing.
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Split data into training and validation sets
)

# Load and split the dataset into training and validation sets.
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Adjust this based on your labeling format
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Adjust this based on your labeling format
    subset='validation'
)

# # Define your CNN model.
# model = keras.Sequential([
#     # Add convolutional and pooling layers here.
#     # Don't forget to add a Flatten and Dense layer for classification.
# ])
model = keras.Sequential([
    # Convolutional layer 1
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Convolutional layer 2
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten the feature maps
    keras.layers.Flatten(),
    
    # Dense layers for classification
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # Change num_classes to the number of sign language classes you have
])

# Compile the model.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model.
epochs = 10  # Adjust the number of training epochs as needed
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Evaluate the model.
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test accuracy: {test_accuracy}")

# Save the trained model for later use.
model.save("sign_language_detection_model.h5")
