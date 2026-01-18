import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = "data/raw/cover_data.csv"
OUTPUT_PATH = "data/processed/"

# Load data
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Convert labels from 1-7 to 0-6 (required for Keras)
y = y - 1

# Train / validation / test split 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save processed arrays
np.save(OUTPUT_PATH + "X_train.npy", X_train_scaled)
np.save(OUTPUT_PATH + "X_val.npy", X_val_scaled)
np.save(OUTPUT_PATH + "X_test.npy", X_test_scaled)

np.save(OUTPUT_PATH + "y_train.npy", y_train.to_numpy())
np.save(OUTPUT_PATH + "y_val.npy", y_val.to_numpy())
np.save(OUTPUT_PATH + "y_test.npy", y_test.to_numpy())

print("Preprocessing complete!")
print("Train shape:", X_train_scaled.shape)
print("Valodation shape:", X_val_scaled.shape)
print("Test shape:", X_test_scaled.shape)
