from env import *
import os
import tensorflow as tf
import visualkeras
from PIL import ImageFont

MODELS_FOLDER = WEIGHTS_PATH
OUTPUT_FOLDER = ARCH_PATH

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited! (just joking)
# Get all .keras files in directory
model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith(".keras")]

for model_file in model_files:
    path = os.path.join(MODELS_FOLDER, model_file)
    print(f"Processing: {model_file}")

    # Load model
    model = tf.keras.models.load_model(path)
    model.summary()
    model.build((32, 224, 224, 3))  # Adjust input shape as necessary

    # Create visualkeras diagram
    try:
        for layer in model.layers:
            if not hasattr(layer, "output_shape"):
                try:
                    layer.output_shape = tuple(layer.output.shape)
                except Exception:
                    pass
        img = visualkeras.layered_view(
            model,
            legend=True,
            font=font,
            sizing_mode='balanced',
            max_xy=200,
            to_file=model_file.replace(".keras", ".png")
        )
    except Exception as e:
        print(f"Failed to visualize {model_file}: {e}")
        continue