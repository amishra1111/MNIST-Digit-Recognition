"""
train_and_export.py

Trains a small CNN digit classifier (0-9) on MNIST and exports it as a
TensorFlow.js model that the Next.js web app can load and run entirely
in the browser (no backend needed at all).

RUN THIS ON YOUR OWN MACHINE (not in a serverless/Vercel environment).
It needs internet access to download MNIST the first time.

Usage:
    pip install tensorflow tensorflowjs scikit-learn numpy
    python train_and_export.py

Output:
    ../web/public/model/model.json
    ../web/public/model/group1-shard1of1.bin
    (TF.js loads these automatically at runtime from /model/model.json)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs
import os

print("Fetching MNIST (this may take a minute the first time)...")

# Keras ships MNIST directly - much faster and more reliable than fetch_openml
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dimension for Conv2D
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# A small CNN - accurate (~99% test accuracy) but tiny enough to run
# instantly in a browser via TensorFlow.js
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("Training...")
model.fit(
    x_train, y_train_cat,
    epochs=8,
    batch_size=128,
    validation_split=0.1,
    verbose=2,
)

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Export straight into the web app's public folder so it's served as a
# static asset and loaded client-side with tf.loadLayersModel()
output_dir = os.path.join("..", "web", "public", "model")
os.makedirs(output_dir, exist_ok=True)
tfjs.converters.save_keras_model(model, output_dir)

print(f"\nModel exported to {output_dir}")
print("You can now run the web app: cd ../web && npm install && npm run dev")
