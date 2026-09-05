import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import *
from src.model import create_model
from tensorflow.keras.preprocessing import image

def predict_corrosion(img_path):
    model = create_model(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.load_weights(MODEL_SAVE_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    class_labels = ['non_corroded', 'corroded']   # adjust order to match training

    return class_labels[class_idx], float(confidence)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()

    label, conf = predict_corrosion(args.image)
    print(f"Prediction: {label} (confidence: {conf:.4f})")