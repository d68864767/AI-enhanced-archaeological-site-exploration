```python
# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from evaluation import load_trained_model, preprocess_data

def plot_training_history(history):
    """
    Function to plot the training history
    :param history: History, history object from model.fit
    """
    # Plot the training loss and accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix():
    """
    Function to plot the confusion matrix
    """
    # Load the trained model
    model = load_trained_model()

    # Preprocess the data
    _, validation_generator = preprocess_data()

    # Predict the classes
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(validation_generator.classes, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()
```
