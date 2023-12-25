```python
# evaluation.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import preprocess_data
from config import MODEL_SAVE_PATH

def load_trained_model():
    """
    Function to load the trained model
    :return: keras.Model, the loaded model
    """
    model_path = MODEL_SAVE_PATH + '/final_model.h5'
    model = load_model(model_path)
    return model

def evaluate_model():
    """
    Function to evaluate the model
    """
    # Load the trained model
    model = load_trained_model()

    # Preprocess the data
    _, validation_generator = preprocess_data()

    # Evaluate the model
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Predict the classes
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print the classification report
    print('Classification Report:')
    target_names = list(validation_generator.class_indices.keys())
    print(classification_report(validation_generator.classes, predicted_classes, target_names=target_names))

    # Print the confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(validation_generator.classes, predicted_classes))

if __name__ == "__main__":
    evaluate_model()
```
