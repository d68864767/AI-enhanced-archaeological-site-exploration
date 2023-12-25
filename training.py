```python
# training.py

import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, SAVE_PLOTS
from data_preprocessing import preprocess_data
from model import create_model
from visualization import plot_training_history

def get_num_classes(train_generator):
    return len(train_generator.class_indices)

def train_model():
    # Preprocess the data
    train_generator, validation_generator = preprocess_data()

    # Get the number of classes
    num_classes = get_num_classes(train_generator)

    # Create the model
    model = create_model(num_classes)

    # Define the callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(MODEL_SAVE_PATH, 'best_model.h5'), save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=10)
    ]

    # Train the model
    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        callbacks=callbacks)

    # Save the final model
    model.save(os.path.join(MODEL_SAVE_PATH, 'final_model.h5'))

    # Plot the training history
    if SAVE_PLOTS:
        plot_training_history(history)

if __name__ == "__main__":
    train_model()
```
