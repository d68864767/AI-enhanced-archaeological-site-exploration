```python
# data_preprocessing.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_PATH, IMAGE_SIZE, TEST_SPLIT, BATCH_SIZE

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        class_folder = os.path.join(folder, filename)
        if os.path.isdir(class_folder):
            for image_file in os.listdir(class_folder):
                images.append(os.path.join(class_folder, image_file))
                labels.append(filename)
    return images, labels

def split_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels)
    return X_train, X_test, y_train, y_test

def create_data_generators(X_train, X_test, y_train, y_test):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(X_train,
                                                        target_size=IMAGE_SIZE,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(X_test,
                                                            target_size=IMAGE_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='categorical')

    return train_generator, validation_generator

def preprocess_data():
    images, labels = load_images_from_folder(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(images, labels)
    train_generator, validation_generator = create_data_generators(X_train, X_test, y_train, y_test)
    return train_generator, validation_generator
```
