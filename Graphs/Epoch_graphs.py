from env import *
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot(df, name):
    acc = df["accuracy"].tolist()
    val_acc = df["val_accuracy"].tolist()
    loss = df["loss"].tolist()
    val_loss = df["val_loss"].tolist()

    epochs = list(range(1, len(acc) + 1))

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.title('Training and Validation Loss')
    
    plt.xlabel('epoch')

    plot_path = os.path.join(GRAPH_PATH, name + '.png')
    plt.savefig(plot_path)
    print(f"Saved training plot to {plot_path}")

csv_files = [
    f for f in os.listdir(LOG_PATH)
    if f.endswith(".csv")
]

if not csv_files:
    raise RuntimeError("No CSV files found in directory")

for csv_file in csv_files:
    path = os.path.join(LOG_PATH, csv_file)
    df = pd.read_csv(path)
    plot(df, os.path.splitext(csv_file)[0])
