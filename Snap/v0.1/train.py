import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.data_preprocessing import split_dataset, get_data_generators
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    # 1. Split raw data into train/val if not already done
    processed_train_dir = os.path.join(PROCESSED_DIR, 'train')
    processed_val_dir = os.path.join(PROCESSED_DIR, 'val')
    if not os.path.exists(processed_train_dir):
        print("Splitting dataset...")
        split_dataset(DATA_DIR, PROCESSED_DIR, train_size=TRAIN_SPLIT)

    # 2. Data generators
    train_gen, val_gen = get_data_generators(processed_train_dir, processed_val_dir,
                                             IMG_SIZE, BATCH_SIZE)

    # 3. Model
    model = create_model(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES,
                         learning_rate=LEARNING_RATE)
    model.summary()

    # 4. Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 5. Train
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stop]
    )

    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()