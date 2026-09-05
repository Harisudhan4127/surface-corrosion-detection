import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_dataset(data_dir, output_dir, train_size=0.8):
    """
    Splits the raw dataset into train/ and val/ directories.
    Expects data_dir/corroded/ and data_dir/non_corroded/
    """
    for class_name in ['corroded', 'non_corroded']:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Directory {class_path} not found.")
        images = os.listdir(class_path)
        train_imgs, val_imgs = train_test_split(images, train_size=train_size, random_state=42)

        for phase, img_list in [('train', train_imgs), ('val', val_imgs)]:
            dest_dir = os.path.join(output_dir, phase, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)
                shutil.copy2(src, dst)
    print("Dataset split completed.")

def get_data_generators(train_dir, val_dir, img_size, batch_size):
    """
    Creates training and validation data generators with augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, val_generator