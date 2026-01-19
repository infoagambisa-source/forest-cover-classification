import numpy as np
import tensorflow as tf
import tensorflow as keras
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Load trained model
model = tf.keras.models.load_model("models/model_v2_weighted.keras")


# Predict probabilities  -> class indices
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("Test set shape:", X_test.shape)
print("\nClassification report (labels 0-6):")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))