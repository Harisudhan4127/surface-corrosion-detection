import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "corrosion_cnn_model.h5")

# Dataset
IMG_SIZE = (224, 224)       # MobileNetV2 expects 224x224
BATCH_SIZE = 32
NUM_CLASSES = 2              # corroded, non_corroded
TRAIN_SPLIT = 0.8            # 80% train, 20% validation (test taken from validation for simplicity)

# Training hyperparameters
EPOCHS = 20
LEARNING_RATE = 1e-4