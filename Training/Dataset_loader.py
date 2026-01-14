from env import *
import os
import tensorflow as tf

def dataset_loader(batch_size=32, img_size=(224, 224)):

    # --- Dataset paths ---
    train_dir = os.path.join(PATH, 'train')
    val_dir = os.path.join(PATH, 'val')

    # --- Load datasets ---
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        class_names=["normal", "abnormal"],
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='binary',
        class_names=["normal", "abnormal"],
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    return train_dataset, val_dataset