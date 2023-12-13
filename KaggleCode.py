
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Path to your dataset
train_data_path = '/Users/jp_ferzacca/Intro_to_AI/Kaggle_Project_HMW5/histopathologic-cancer-detection/train'
labels_csv_path = '/Users/jp_ferzacca/Intro_to_AI/Kaggle_Project_HMW5/histopathologic-cancer-detection/train_labels.csv'
test_data_path = '/Users/jp_ferzacca/Intro_to_AI/Kaggle_Project_HMW5/histopathologic-cancer-detection/test' # Correct path

# Load and prepare labels
labels_df = pd.read_csv(labels_csv_path)
labels_df['id'] = labels_df['id'].apply(lambda x: f"{x}.tif")
labels_df['file_path'] = labels_df['id'].apply(lambda x: os.path.join(train_data_path, x))
labels_df['label'] = labels_df['label'].astype(str)  # Convert labels to strings for Keras

# Splitting the dataset into training and validation sets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Image preprocessing/augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation data

# Flow data from dataframe for training
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',
    y_col='label',
    target_size=(96, 96),  # Adjust this if the image size is different
    batch_size=32,
    class_mode='binary'
)

# Flow data from dataframe for validation
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',
    y_col='label',
    target_size=(96, 96),  # Adjust this if the image size is different
    batch_size=32,
    class_mode='binary'
)

# Now, train_generator and val_generator are ready to be used in a model.fit call



# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),  # First convolutional layer
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'), # Third convolutional layer
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout layer to reduce overfitting
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer
              loss='binary_crossentropy',  # Binary cross-entropy loss function
              metrics=['accuracy'])  # Accuracy metric

print("Starting model training...")
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=10,  # Number of epochs can be adjusted
    validation_data=val_generator,
    validation_steps=val_generator.n // val_generator.batch_size
)
print("Model training completed.")
# Save the trained model (optional)
model.save('/Users/jp_ferzacca/Intro_to_AI/Kaggle_Project_HMW5/my_trained_model.h5')


# Create a DataFrame for the test data
test_images = [f for f in os.listdir(test_data_path) if f.endswith('.tif')]
test_df = pd.DataFrame({'id': test_images, 'file_path': test_images})
test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(test_data_path, x))

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='file_path',
    y_col=None,
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Predict
predictions = model.predict(test_generator, steps=len(test_generator))

# Flatten to 1D array
predictions = predictions.flatten()

# Convert predictions to binary (0 or 1) based on a threshold, e.g., 0.5
predictions_binary = (predictions > 0.5).astype(int)

# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'label': predictions_binary})

# Remove file extension from 'id' column
submission_df['id'] = submission_df['id'].apply(lambda x: x.split('.')[0])

# Save submission file
submission_df.to_csv('submission.csv', index=False)
