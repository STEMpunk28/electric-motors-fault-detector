from env import *
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------

MODELS_DIR = WEIGHTS_PATH
TEST_DIR   = os.path.join(PATH, 'test')
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
THRESHOLD  = 0.5

# ---------------------------------------------------------
# LOAD TEST DATASET
# ---------------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "abnormal"],
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

X_test, Y_test = [], []

for imgs, labels in test_ds:
    X_test.append(imgs.numpy())
    Y_test.append(labels.numpy())

X_test = np.concatenate(X_test)
Y_test = np.concatenate(Y_test).astype(int)

print(f"Loaded {len(X_test)} test images.")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_keras_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        raise RuntimeError(f"Could not load model {path}: {e}")

def predict_model(model, x):
    y_prob = model.predict(x, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return y_pred

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()

    save_path = os.path.join(
        GRAPH_PATH,
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix to {save_path}")

# ---------------------------------------------------------
# EVALUATE ALL MODELS
# ---------------------------------------------------------

for file in os.listdir(MODELS_DIR):
    if not file.endswith(".keras"):
        continue

    model_path = os.path.join(MODELS_DIR, file)
    model_name = os.path.splitext(file)[0]

    print(f"\nEvaluating: {file}")

    model = load_keras_model(model_path)
    y_pred = predict_model(model, X_test)

    plot_confusion_matrix(
        y_true=Y_test,
        y_pred=y_pred,
        model_name=model_name
    )
