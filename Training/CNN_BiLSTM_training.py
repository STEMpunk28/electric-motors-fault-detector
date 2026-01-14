from env import *
from Dataset_loader import dataset_loader
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# --- Parameters ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE

train_dataset, val_dataset = dataset_loader(BATCH_SIZE, IMG_SIZE)

train_dataset = train_dataset.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))
val_dataset = val_dataset.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))
train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# Model layers
inputs = layers.Input(shape=(224, 224, 3))

base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False
x = base(inputs, training=False)  # (7, 7, 1280)

x = layers.Permute((2,1,3))(x)
x = layers.Conv2D(256, 1, activation='relu')(x)
x = layers.Reshape((7, 7*256))(x)
x = layers.TimeDistributed(layers.Dense(512, activation='relu'))(x)
x = layers.Bidirectional(layers.LSTM(256))(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model([inputs], [outputs], name='CNN_BiLSTM_EfficientNet')
model.summary()

# --- Compile ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
checkpoint_path = os.path.join(WEIGHTS_PATH, 'CNN_BiLSTM_training.keras')
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    'CNN_BiLSTM_training.csv',
    append=True
)

callbacks = [checkpoint_cb, reduce_lr_cb, earlystop_cb, csv_logger]

# --- Train ---
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(val_dataset, verbose=2)
print(f'Validation loss: {test_loss:.4f}')
print(f'Validation accuracy: {test_acc:.4f}')