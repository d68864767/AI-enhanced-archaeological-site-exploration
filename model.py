```python
# model.py

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from config import IMAGE_SIZE, IMAGE_CHANNELS, MODEL_ARCHITECTURE, PRETRAINED, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY

def create_model(num_classes):
    """
    Function to create a model
    :param num_classes: int, number of classes
    :return: keras.Model, the created model
    """
    # Define the base model
    if MODEL_ARCHITECTURE == 'resnet50':
        base_model = ResNet50(include_top=False, weights='imagenet' if PRETRAINED else None,
                              input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS), pooling='avg')
    else:
        raise ValueError(f"Invalid model architecture '{MODEL_ARCHITECTURE}'")

    # Create a new model on top of the base model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=WEIGHT_DECAY),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model
```
