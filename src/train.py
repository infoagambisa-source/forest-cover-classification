import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility to help compare runs
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load preprocessed data
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

num_features = X_train.shape[1]
num_classes = 7 

# Building a simple baseline model (MLP)
model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Early stopping to stop training when val loss stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=512,
    callbacks=[early_stop],
    verbose=2
)

# Save model
model.save("models/model_v1.keras")
print("Saved model to models/model_v1.keras")