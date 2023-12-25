```python
# utils.py

import os
import matplotlib.pyplot as plt
from config import MODEL_SAVE_PATH, SAVE_PLOTS

def save_model(model):
    """
    Function to save a trained model
    :param model: trained model
    """
    model.save(MODEL_SAVE_PATH)

def plot_training_history(history):
    """
    Function to plot training history
    :param history: training history
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Save the plot to the disk if SAVE_PLOTS is True
    if SAVE_PLOTS:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/training_plot.png')

    plt.show()

def calculate_class_weights(y_train):
    """
    Function to calculate class weights
    :param y_train: training labels
    :return: class weights
    """
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return class_weights
```
