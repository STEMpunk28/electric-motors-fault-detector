import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
from env import *

from tensorflow.keras.preprocessing import image_dataset_from_directory

# ---------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------

MODELS_DIR = WEIGHTS_PATH                  # where all models are stored
TEST_DIR   = os.path.join(PATH, 'test')    # test dataset path
IMG_SIZE   = (224, 224)                    # change to whatever your models expect
BATCH_SIZE = 32
THRESHOLD  = 0.5                           # for binary models

# ---------------------------------------------------------
# LOAD TEST DATASET
# ---------------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Extract numpy arrays for evaluation
X_test = []
Y_test = []

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
    except Exception:
        raise ValueError(f"Could not load Keras model: {path}")

def load_joblib_model(path):
    return joblib.load(path)

def predict_model(model, x):
    y_prob = model.predict(x, verbose=0)   # sigmoid output
    return (y_prob.ravel() >= THRESHOLD).astype(int)

# ---------------------------------------------------------
# EVALUATE ALL MODELS
# ---------------------------------------------------------

results = []

for file in os.listdir(MODELS_DIR):
    path = os.path.join(MODELS_DIR, file)

    if not file.endswith(".keras"):
        print(f"Skipping: {file}")
        continue

    print(f"\nEvaluating: {file}")

    model = load_keras_model(path)
    y_pred = predict_model(model, X_test)

    acc  = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred, zero_division=0)
    f1   = f1_score(Y_test, y_pred)

    print(
        f"{file} → "
        f"Accuracy: {acc:.4f}, "
        f"Precision: {prec:.4f}, "
        f"F1: {f1:.4f}"
    )

    results.append((file.replace(".keras", ""), acc, prec, f1))

# ---------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------

def plot_metric(names, values, metric_name, filename):
    plt.figure(figsize=(12, 6))
    # Generate different colors automatically
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, values, color=colors)
    # Add value labels on top of each bar
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")
    plt.tight_layout()
    path = os.path.join(GRAPH_PATH, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved {metric_name} plot to {path}")

# ---------------------------------------------------------
# GENERATE GRAPHS
# ---------------------------------------------------------

names      = [r[0] for r in results]
accuracies = [r[1] for r in results]
precisions = [r[2] for r in results]
f1_scores  = [r[3] for r in results]

plot_metric(names, accuracies, "Accuracy",  "accuracy_test.png")
plot_metric(names, precisions, "Precision", "precision_test.png")
plot_metric(names, f1_scores,  "F1 Score",  "f1_score_test.png")
