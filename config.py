```python
# config.py

# Paths
DATA_PATH = 'path/to/your/dataset'
MODEL_SAVE_PATH = 'path/to/save/your/model'

# Image Processing
IMAGE_SIZE = (224, 224)  # Size of the input images
IMAGE_CHANNELS = 3  # Number of channels of the input images

# Model Parameters
MODEL_ARCHITECTURE = 'resnet50'  # Model architecture to be used
PRETRAINED = True  # Whether to use pretrained weights

# Training Parameters
BATCH_SIZE = 32  # Number of samples per gradient update
EPOCHS = 100  # Number of times to iterate over the entire dataset
LEARNING_RATE = 0.001  # Learning rate for the optimizer
MOMENTUM = 0.9  # Momentum for the optimizer
WEIGHT_DECAY = 0.0005  # Weight decay for the optimizer

# Evaluation Parameters
TEST_SPLIT = 0.2  # Fraction of data to be used for testing

# Visualization Parameters
SAVE_PLOTS = True  # Whether to save plots during training
```
