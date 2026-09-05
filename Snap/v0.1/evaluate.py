import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import *
from src.model import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Load model
    model = create_model(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.load_weights(MODEL_SAVE_PATH)

    # Use validation set as test set (for demonstration)
    test_dir = os.path.join(PROCESSED_DIR, 'val')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Predict
    predictions = model.predict(test_gen, steps=test_gen.samples // BATCH_SIZE + 1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes[:len(y_pred)]

    # Report
    target_names = list(test_gen.class_indices.keys())
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()