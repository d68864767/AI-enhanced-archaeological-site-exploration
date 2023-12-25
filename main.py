```python
# main.py

# Import necessary modules
import config
from data_preprocessing import preprocess_data
from image_processing import process_images
from model import create_model
from training import train_model
from evaluation import evaluate_model
from visualization import visualize_results
from utils import load_data, split_data

def main():
    # Load and preprocess data
    data = load_data(config.DATA_PATH)
    data = preprocess_data(data)

    # Process images
    images = process_images(data, config.IMAGE_SIZE, config.IMAGE_CHANNELS)

    # Split data into training and testing sets
    train_data, test_data = split_data(images, config.TEST_SPLIT)

    # Create model
    model = create_model(config.MODEL_ARCHITECTURE, config.PRETRAINED)

    # Train model
    model = train_model(model, train_data, config.BATCH_SIZE, config.EPOCHS, 
                        config.LEARNING_RATE, config.MOMENTUM, config.WEIGHT_DECAY)

    # Evaluate model
    results = evaluate_model(model, test_data)

    # Visualize results
    if config.SAVE_PLOTS:
        visualize_results(results)

    # Save model
    model.save(config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
```
